"""
Tests for project_crawler.py — Phase 3, Component 1.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from hce_core import EdgeType, EntityGraph, NodeType
from project_crawler import (
    CrawlConfig,
    ParseResult,
    _ALL_EXTENSIONS,
    _LANGUAGE_MAP,
    crawl_project,
)

# Absolute path to the HCE project root (used for self-crawl tests).
HCE_ROOT = Path(__file__).resolve().parent


# ── Helpers ──────────────────────────────────────────────────────────────

def _edge_types_between(graph: EntityGraph, src: str, tgt: str) -> list[str]:
    """Return a list of edge_type values for all edges from *src* to *tgt*."""
    types: list[str] = []
    for nbr, edata in graph.get_neighbors(src, direction="out"):
        if nbr == tgt:
            types.append(edata["edge_type"])
    return types


# ── 1. Self-crawl: find FILE and FUNCTION nodes, and edges ──────────────

class TestSelfCrawl:
    """Crawl this very HCE project and verify structural expectations."""

    @pytest.fixture(scope="class")
    def graph(self) -> EntityGraph:
        return crawl_project(HCE_ROOT)

    def test_finds_py_files(self, graph: EntityGraph) -> None:
        file_ids = graph.get_nodes_by_type(NodeType.FILE)
        # At minimum we expect hce_core.py, project_crawler.py, and this test file.
        names = {Path(fid).name for fid in file_ids}
        assert "hce_core.py" in names
        assert "project_crawler.py" in names
        assert "test_project_crawler.py" in names

    def test_finds_function_nodes(self, graph: EntityGraph) -> None:
        func_ids = graph.get_nodes_by_type(NodeType.FUNCTION)
        labels = {graph.get_node(fid)["label"] for fid in func_ids}
        # EntityGraph class and spreading_activation function should be found.
        assert "EntityGraph" in labels
        assert "spreading_activation" in labels
        # The crawler's own entry point too.
        assert "crawl_project" in labels

    def test_imports_edges_exist(self, graph: EntityGraph) -> None:
        # project_crawler.py imports hce_core (should resolve to the file node).
        edge_types = _edge_types_between(graph, "project_crawler.py", "hce_core.py")
        assert EdgeType.IMPORTS.value in edge_types

    def test_function_part_of_file(self, graph: EntityGraph) -> None:
        # crawl_project is PART_OF project_crawler.py
        node_id = "project_crawler.py::crawl_project"
        edge_types = _edge_types_between(graph, node_id, "project_crawler.py")
        assert EdgeType.PART_OF.value in edge_types


# ── 2. CrawlConfig ignore patterns ──────────────────────────────────────

class TestIgnorePatterns:
    def test_default_ignores_pycache(self) -> None:
        """__pycache__ dirs should be skipped with default config."""
        graph = crawl_project(HCE_ROOT)
        file_ids = graph.get_nodes_by_type(NodeType.FILE)
        for fid in file_ids:
            assert "__pycache__" not in fid

    def test_custom_ignore_skips_tests(self) -> None:
        """A custom ignore pattern can exclude test files."""
        config = CrawlConfig(ignore_patterns=["test_*", "__pycache__", ".git", ".venv", ".pytest_cache"])
        graph = crawl_project(HCE_ROOT, config=config)
        file_ids = graph.get_nodes_by_type(NodeType.FILE)
        names = {Path(fid).name for fid in file_ids}
        assert "test_project_crawler.py" not in names
        # Non-test files should still be present.
        assert "hce_core.py" in names


# ── 3. PART_OF edges ────────────────────────────────────────────────────

class TestPartOfEdges:
    def test_every_function_has_part_of(self) -> None:
        """Every FUNCTION node should have at least one PART_OF edge."""
        graph = crawl_project(HCE_ROOT)
        func_ids = graph.get_nodes_by_type(NodeType.FUNCTION)
        for fid in func_ids:
            neighbors = graph.get_neighbors(fid, edge_type=EdgeType.PART_OF, direction="out")
            assert len(neighbors) >= 1, f"{fid} has no PART_OF edge"

    def test_part_of_target_is_file(self) -> None:
        """The target of every PART_OF edge should be a FILE node."""
        graph = crawl_project(HCE_ROOT)
        func_ids = graph.get_nodes_by_type(NodeType.FUNCTION)
        for fid in func_ids:
            for target, _ in graph.get_neighbors(fid, edge_type=EdgeType.PART_OF, direction="out"):
                node = graph.get_node(target)
                assert node is not None
                assert node["node_type"] == NodeType.FILE.value


# ── 4. IMPORTS edges ────────────────────────────────────────────────────

class TestImportsEdges:
    def test_external_module_creates_concept_node(self) -> None:
        """Importing a stdlib/external module should produce a CONCEPT node."""
        graph = crawl_project(HCE_ROOT)
        # hce_core.py imports networkx -> CONCEPT node "networkx"
        assert graph.has_node("networkx")
        node = graph.get_node("networkx")
        assert node is not None
        assert node["node_type"] == NodeType.CONCEPT.value

    def test_internal_import_resolves_to_file(self) -> None:
        """Importing a project module should resolve to the FILE node."""
        graph = crawl_project(HCE_ROOT)
        edge_types = _edge_types_between(graph, "project_crawler.py", "hce_core.py")
        assert EdgeType.IMPORTS.value in edge_types


# ── 5. Graceful handling of syntax errors ────────────────────────────────

class TestSyntaxErrorHandling:
    def test_bad_syntax_file_skipped(self, tmp_path: Path) -> None:
        """A file with a syntax error should be skipped (not crash)."""
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def oops(:\n    pass\n")

        good_file = tmp_path / "good.py"
        good_file.write_text("def hello():\n    return 1\n")

        graph = crawl_project(tmp_path)
        file_ids = graph.get_nodes_by_type(NodeType.FILE)
        names = {Path(fid).name for fid in file_ids}

        # Both files should appear as FILE nodes (the file is read before parsing).
        # But only 'good.py' should have FUNCTION children.
        assert "good.py" in names
        assert "bad.py" in names

        func_ids = graph.get_nodes_by_type(NodeType.FUNCTION)
        labels = {graph.get_node(f)["label"] for f in func_ids}
        assert "hello" in labels
        # "oops" should NOT appear because the file failed to parse.
        assert "oops" not in labels


# ── 6. Empty directory ──────────────────────────────────────────────────

class TestEmptyDirectory:
    def test_empty_dir_returns_empty_graph(self, tmp_path: Path) -> None:
        """Crawling an empty directory should return an empty graph."""
        graph = crawl_project(tmp_path)
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_dir_with_no_matching_files(self, tmp_path: Path) -> None:
        """A directory with only unsupported files should return an empty graph."""
        (tmp_path / "readme.md").write_text("# Hello")
        (tmp_path / "data.json").write_text("{}")
        graph = crawl_project(tmp_path)
        assert graph.node_count == 0
        assert graph.edge_count == 0


# ── 7. Default extensions include all supported languages ────────────────

class TestDefaultExtensions:
    def test_default_includes_all_supported(self) -> None:
        config = CrawlConfig()
        for ext in _ALL_EXTENSIONS:
            assert ext in config.extensions, f"Default config missing {ext}"

    def test_all_extensions_have_parser(self) -> None:
        for ext in _ALL_EXTENSIONS:
            assert ext in _LANGUAGE_MAP, f"No parser for {ext}"

    def test_default_ignores_build_dirs(self) -> None:
        config = CrawlConfig()
        for d in ("target", "build", "dist", "vendor"):
            assert d in config.ignore_patterns


# ── 8. Java parser ──────────────────────────────────────────────────────

class TestJavaParser:
    SAMPLE = """\
package com.example;

import java.util.List;
import com.google.common.collect.ImmutableList;

public class UserService {
    private final List<User> users;

    public User findById(int id) {
        return users.get(id);
    }

    public static void main(String[] args) {
        System.out.println("hello");
    }
}

public interface Repository {
    void save(Object entity);
}

public enum Status {
    ACTIVE, INACTIVE
}
"""

    def test_creates_file_and_function_nodes(self, tmp_path: Path) -> None:
        (tmp_path / "UserService.java").write_text(self.SAMPLE)
        graph = crawl_project(tmp_path)

        file_ids = graph.get_nodes_by_type(NodeType.FILE)
        assert any("UserService.java" in f for f in file_ids)

        func_ids = graph.get_nodes_by_type(NodeType.FUNCTION)
        labels = {graph.get_node(f)["label"] for f in func_ids}
        assert "UserService" in labels
        assert "Repository" in labels
        assert "Status" in labels
        assert "findById" in labels

    def test_import_concept_nodes(self, tmp_path: Path) -> None:
        (tmp_path / "App.java").write_text(self.SAMPLE)
        graph = crawl_project(tmp_path)
        assert graph.has_node("java.util.List")
        node = graph.get_node("java.util.List")
        assert node["node_type"] == NodeType.CONCEPT.value

    def test_part_of_edges(self, tmp_path: Path) -> None:
        (tmp_path / "UserService.java").write_text(self.SAMPLE)
        graph = crawl_project(tmp_path)
        edge_types = _edge_types_between(
            graph, "UserService.java::UserService", "UserService.java"
        )
        assert EdgeType.PART_OF.value in edge_types


# ── 9. JavaScript / TypeScript parser ───────────────────────────────────

class TestJavaScriptParser:
    SAMPLE_JS = """\
import React from 'react';
import { useState } from 'react';
const axios = require('axios');

export default function App() {
    return <div>Hello</div>;
}

export class UserComponent {
    render() { return null; }
}

export const fetchData = async (url) => {
    return axios.get(url);
};
"""

    SAMPLE_TS = """\
import { Request, Response } from 'express';

export interface ApiResponse {
    status: number;
    data: unknown;
}

export type UserId = string;

export async function handleRequest(req: Request): Promise<ApiResponse> {
    return { status: 200, data: null };
}
"""

    def test_js_definitions(self, tmp_path: Path) -> None:
        (tmp_path / "app.js").write_text(self.SAMPLE_JS)
        graph = crawl_project(tmp_path)

        func_ids = graph.get_nodes_by_type(NodeType.FUNCTION)
        labels = {graph.get_node(f)["label"] for f in func_ids}
        assert "App" in labels
        assert "UserComponent" in labels
        assert "fetchData" in labels

    def test_js_imports(self, tmp_path: Path) -> None:
        (tmp_path / "app.js").write_text(self.SAMPLE_JS)
        graph = crawl_project(tmp_path)
        assert graph.has_node("react")
        assert graph.has_node("axios")

    def test_ts_interface_and_type(self, tmp_path: Path) -> None:
        (tmp_path / "handler.ts").write_text(self.SAMPLE_TS)
        graph = crawl_project(tmp_path)

        func_ids = graph.get_nodes_by_type(NodeType.FUNCTION)
        labels = {graph.get_node(f)["label"] for f in func_ids}
        assert "ApiResponse" in labels
        assert "UserId" in labels
        assert "handleRequest" in labels

    def test_tsx_extension(self, tmp_path: Path) -> None:
        (tmp_path / "comp.tsx").write_text("export function Comp() { return null; }\n")
        graph = crawl_project(tmp_path)
        func_ids = graph.get_nodes_by_type(NodeType.FUNCTION)
        labels = {graph.get_node(f)["label"] for f in func_ids}
        assert "Comp" in labels


# ── 10. Go parser ───────────────────────────────────────────────────────

class TestGoParser:
    SAMPLE = """\
package main

import (
    "fmt"
    "net/http"
)

import "os"

type Server struct {
    Port int
}

type Handler interface {
    ServeHTTP(w http.ResponseWriter, r *http.Request)
}

func NewServer(port int) *Server {
    return &Server{Port: port}
}

func (s *Server) Start() error {
    fmt.Println("starting")
    return nil
}
"""

    def test_go_definitions(self, tmp_path: Path) -> None:
        (tmp_path / "main.go").write_text(self.SAMPLE)
        graph = crawl_project(tmp_path)

        func_ids = graph.get_nodes_by_type(NodeType.FUNCTION)
        labels = {graph.get_node(f)["label"] for f in func_ids}
        assert "NewServer" in labels
        assert "Start" in labels
        assert "Server" in labels
        assert "Handler" in labels

    def test_go_imports(self, tmp_path: Path) -> None:
        (tmp_path / "main.go").write_text(self.SAMPLE)
        graph = crawl_project(tmp_path)
        assert graph.has_node("fmt")
        assert graph.has_node("net/http")
        assert graph.has_node("os")

    def test_go_struct_metadata(self, tmp_path: Path) -> None:
        (tmp_path / "main.go").write_text(self.SAMPLE)
        graph = crawl_project(tmp_path)
        node = graph.get_node("main.go::Server")
        assert node is not None
        assert node["metadata"]["type"] == "struct"


# ── 11. Rust parser ─────────────────────────────────────────────────────

class TestRustParser:
    SAMPLE = """\
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

pub struct Config {
    pub name: String,
    pub port: u16,
}

pub trait Configurable {
    fn configure(&self) -> Config;
}

impl Config {
    pub fn new(name: String, port: u16) -> Self {
        Config { name, port }
    }
}

pub enum Status {
    Active,
    Inactive,
}

pub async fn start_server(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

fn helper() -> bool {
    true
}
"""

    def test_rust_definitions(self, tmp_path: Path) -> None:
        (tmp_path / "lib.rs").write_text(self.SAMPLE)
        graph = crawl_project(tmp_path)

        func_ids = graph.get_nodes_by_type(NodeType.FUNCTION)
        labels = {graph.get_node(f)["label"] for f in func_ids}
        assert "Config" in labels  # struct + impl
        assert "Configurable" in labels
        assert "Status" in labels
        assert "start_server" in labels
        assert "helper" in labels
        assert "new" in labels

    def test_rust_imports(self, tmp_path: Path) -> None:
        (tmp_path / "lib.rs").write_text(self.SAMPLE)
        graph = crawl_project(tmp_path)
        assert graph.has_node("std::collections")
        assert graph.has_node("serde")

    def test_rust_kind_metadata(self, tmp_path: Path) -> None:
        (tmp_path / "lib.rs").write_text(self.SAMPLE)
        graph = crawl_project(tmp_path)
        node = graph.get_node("lib.rs::Configurable")
        assert node is not None
        assert node["metadata"]["type"] == "trait"


# ── 12. C/C++ parser ───────────────────────────────────────────────────

class TestCCppParser:
    SAMPLE_CPP = """\
#include <iostream>
#include "config.h"

namespace MyApp {

class Engine {
public:
    void start();
    int getStatus();
};

struct Settings {
    int port;
    bool debug;
};

void Engine::start() {
    std::cout << "starting" << std::endl;
}

int Engine::getStatus() {
    return 0;
}

}  // namespace MyApp
"""

    def test_cpp_definitions(self, tmp_path: Path) -> None:
        (tmp_path / "engine.cpp").write_text(self.SAMPLE_CPP)
        graph = crawl_project(tmp_path)

        func_ids = graph.get_nodes_by_type(NodeType.FUNCTION)
        labels = {graph.get_node(f)["label"] for f in func_ids}
        assert "Engine" in labels
        assert "Settings" in labels
        assert "MyApp" in labels

    def test_cpp_includes(self, tmp_path: Path) -> None:
        (tmp_path / "engine.cpp").write_text(self.SAMPLE_CPP)
        graph = crawl_project(tmp_path)
        assert graph.has_node("iostream")
        assert graph.has_node("config.h")

    def test_header_extension(self, tmp_path: Path) -> None:
        (tmp_path / "config.h").write_text(
            "#ifndef CONFIG_H\n#define CONFIG_H\nstruct AppConfig { int x; };\n#endif\n"
        )
        graph = crawl_project(tmp_path)
        file_ids = graph.get_nodes_by_type(NodeType.FILE)
        assert any("config.h" in f for f in file_ids)


# ── 13. Ruby parser ────────────────────────────────────────────────────

class TestRubyParser:
    SAMPLE = """\
require 'json'
require_relative 'helpers'

module Authentication
  class User
    def initialize(name)
      @name = name
    end

    def self.find(id)
      # ...
    end

    def authenticate!
      # ...
    end
  end
end

def standalone_helper
  true
end
"""

    def test_ruby_definitions(self, tmp_path: Path) -> None:
        (tmp_path / "auth.rb").write_text(self.SAMPLE)
        graph = crawl_project(tmp_path)

        func_ids = graph.get_nodes_by_type(NodeType.FUNCTION)
        labels = {graph.get_node(f)["label"] for f in func_ids}
        assert "User" in labels
        assert "Authentication" in labels
        assert "initialize" in labels
        assert "find" in labels
        assert "authenticate!" in labels
        assert "standalone_helper" in labels

    def test_ruby_imports(self, tmp_path: Path) -> None:
        (tmp_path / "auth.rb").write_text(self.SAMPLE)
        graph = crawl_project(tmp_path)
        assert graph.has_node("json")
        assert graph.has_node("helpers")

    def test_ruby_module_kind(self, tmp_path: Path) -> None:
        (tmp_path / "auth.rb").write_text(self.SAMPLE)
        graph = crawl_project(tmp_path)
        node = graph.get_node("auth.rb::Authentication")
        assert node is not None
        assert node["metadata"]["type"] == "module"


# ── 14. Mixed-language project ──────────────────────────────────────────

class TestMixedLanguageProject:
    def test_all_languages_indexed(self, tmp_path: Path) -> None:
        """A project with one file per language should index all of them."""
        (tmp_path / "app.py").write_text("def main(): pass\n")
        (tmp_path / "App.java").write_text("public class App { void run() {} }\n")
        (tmp_path / "app.js").write_text("export function init() {}\n")
        (tmp_path / "app.ts").write_text("export function setup(): void {}\n")
        (tmp_path / "main.go").write_text("package main\nfunc Main() {}\n")
        (tmp_path / "lib.rs").write_text("pub fn start() {}\n")
        (tmp_path / "engine.cpp").write_text('#include <stdio.h>\nstruct Engine { int x; };\n')
        (tmp_path / "app.rb").write_text("class App; end\n")

        graph = crawl_project(tmp_path)

        file_ids = graph.get_nodes_by_type(NodeType.FILE)
        file_names = {Path(f).name for f in file_ids}
        assert "app.py" in file_names
        assert "App.java" in file_names
        assert "app.js" in file_names
        assert "app.ts" in file_names
        assert "main.go" in file_names
        assert "lib.rs" in file_names
        assert "engine.cpp" in file_names
        assert "app.rb" in file_names

        func_ids = graph.get_nodes_by_type(NodeType.FUNCTION)
        labels = {graph.get_node(f)["label"] for f in func_ids}
        # At least one definition from each language
        assert "main" in labels       # Python
        assert "init" in labels       # JS
        assert "setup" in labels      # TS
        assert "Main" in labels       # Go
        assert "start" in labels      # Rust
        assert "Engine" in labels     # C++


# ── 15. Python parser unchanged (self-crawl with .py filter) ────────────

class TestPythonParserUnchanged:
    """Ensure the refactored Python parser produces the same results as before."""

    @pytest.fixture(scope="class")
    def graph(self) -> EntityGraph:
        return crawl_project(HCE_ROOT, CrawlConfig(extensions=[".py"]))

    def test_finds_py_files(self, graph: EntityGraph) -> None:
        file_ids = graph.get_nodes_by_type(NodeType.FILE)
        names = {Path(fid).name for fid in file_ids}
        assert "hce_core.py" in names
        assert "project_crawler.py" in names

    def test_finds_function_nodes(self, graph: EntityGraph) -> None:
        func_ids = graph.get_nodes_by_type(NodeType.FUNCTION)
        labels = {graph.get_node(fid)["label"] for fid in func_ids}
        assert "EntityGraph" in labels
        assert "spreading_activation" in labels
        assert "crawl_project" in labels

    def test_imports_edges_exist(self, graph: EntityGraph) -> None:
        edge_types = _edge_types_between(graph, "project_crawler.py", "hce_core.py")
        assert EdgeType.IMPORTS.value in edge_types

    def test_function_part_of_file(self, graph: EntityGraph) -> None:
        node_id = "project_crawler.py::crawl_project"
        edge_types = _edge_types_between(graph, node_id, "project_crawler.py")
        assert EdgeType.PART_OF.value in edge_types
