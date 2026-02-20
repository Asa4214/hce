"""
HCE Phase 3, Component 1: Project Crawler.

Walks a codebase and populates an EntityGraph with FILE nodes, FUNCTION
nodes (functions, classes, structs, traits, …), and IMPORTS / CALLS /
PART_OF edges.

Supported languages:
  Python (.py)          — full AST parsing via ``ast`` module
  Java (.java)          — regex-based
  JavaScript/TypeScript (.js, .jsx, .ts, .tsx) — regex-based
  Go (.go)              — regex-based
  Rust (.rs)            — regex-based
  C/C++ (.c, .h, .cpp, .hpp, .cc) — regex-based
  Ruby (.rb)            — regex-based
"""

from __future__ import annotations

import ast
import logging
import re
import sys
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable

from hce_core import EdgeType, EntityGraph, NodeType

log = logging.getLogger(__name__)


# ── Parse Result (common intermediate representation) ─────────────────────

@dataclass
class ParseResult:
    """Language-agnostic output produced by every parser.

    * **definitions** — ``(name, lineno, kind)`` tuples for each
      class / function / struct / trait / … found in the file.
    * **imports** — module or package names that the file imports.
    * **calls** — ``(func_node_id, [called_names])`` pairs so that
      the second-pass call resolver can wire CALLS edges.
    """

    definitions: list[tuple[str, int, str]] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    calls: list[tuple[str, list[str]]] = field(default_factory=list)


# ── Python parser (AST) ──────────────────────────────────────────────────

def _parse_python(source: str, file_id: str) -> ParseResult:
    """Parse a Python source file using the ``ast`` module."""
    try:
        tree = ast.parse(source, filename=file_id)
    except SyntaxError as exc:
        log.warning("Syntax error in %s: %s", file_id, exc)
        return ParseResult()

    result = ParseResult()
    result.imports = _extract_imports(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            node_id = f"{file_id}::{name}"

            if isinstance(node, ast.AsyncFunctionDef):
                kind = "async_function"
            elif isinstance(node, ast.ClassDef):
                kind = "class"
            else:
                kind = "function"

            result.definitions.append((name, node.lineno, kind))

            # Collect call names inside this definition's body.
            calls: list[str] = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    called = _resolve_call_name(child)
                    if called:
                        calls.append(called)
            result.calls.append((node_id, calls))

    return result


def _resolve_call_name(call_node: ast.Call) -> str | None:
    """Best-effort extraction of the called function name from an
    ``ast.Call`` node.  Returns the simple name for ``foo()`` and the
    attribute name for ``obj.foo()``; returns ``None`` for anything more
    complex."""
    func = call_node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _extract_imports(tree: ast.Module) -> list[str]:
    """Return a list of top-level module names imported by *tree*."""
    modules: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    return modules


# ── Java parser (regex) ──────────────────────────────────────────────────

_JAVA_DEF_RE = re.compile(
    r"^\s*(?:public|protected|private|static|abstract|final|synchronized|native|strictfp|\s)*"
    r"(?:class|interface|enum|record)\s+(\w+)",
    re.MULTILINE,
)
_JAVA_METHOD_RE = re.compile(
    r"^\s*(?:public|protected|private|static|abstract|final|synchronized|native|\s)*"
    r"(?:<[^>]+>\s+)?"              # optional generic type params
    r"[\w\[\]<>,\s]+\s+(\w+)\s*\(",  # return-type name(
    re.MULTILINE,
)
_JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+);", re.MULTILINE)


def _parse_java(source: str, file_id: str) -> ParseResult:
    result = ParseResult()

    for m in _JAVA_DEF_RE.finditer(source):
        name = m.group(1)
        lineno = source[: m.start()].count("\n") + 1
        # Determine kind from matched text
        text = m.group(0)
        if "interface" in text:
            kind = "interface"
        elif "enum" in text:
            kind = "enum"
        elif "record" in text:
            kind = "record"
        else:
            kind = "class"
        result.definitions.append((name, lineno, kind))

    for m in _JAVA_METHOD_RE.finditer(source):
        name = m.group(1)
        # Skip false positives that are actually class/interface/enum keywords
        if name in ("class", "interface", "enum", "record", "if", "for",
                     "while", "switch", "catch", "return", "new", "throw"):
            continue
        lineno = source[: m.start()].count("\n") + 1
        result.definitions.append((name, lineno, "method"))

    for m in _JAVA_IMPORT_RE.finditer(source):
        result.imports.append(m.group(1))

    return result


# ── JavaScript / TypeScript parser (regex) ───────────────────────────────

_JS_FUNC_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)",
    re.MULTILINE,
)
_JS_CLASS_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)",
    re.MULTILINE,
)
_JS_ARROW_RE = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[a-zA-Z_]\w*)\s*=>",
    re.MULTILINE,
)
_JS_TS_IFACE_RE = re.compile(
    r"^\s*(?:export\s+)?(?:interface|type)\s+(\w+)",
    re.MULTILINE,
)
_JS_IMPORT_FROM_RE = re.compile(
    r"""(?:import\s+.*?\s+from\s+['"]([^'"]+)['"]|require\s*\(\s*['"]([^'"]+)['"]\s*\))""",
    re.MULTILINE,
)


def _parse_javascript(source: str, file_id: str) -> ParseResult:
    result = ParseResult()

    for m in _JS_FUNC_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        kind = "async_function" if "async" in m.group(0) else "function"
        result.definitions.append((m.group(1), lineno, kind))

    for m in _JS_CLASS_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        result.definitions.append((m.group(1), lineno, "class"))

    for m in _JS_ARROW_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        kind = "async_function" if "async" in m.group(0) else "function"
        result.definitions.append((m.group(1), lineno, kind))

    for m in _JS_TS_IFACE_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        text = m.group(0)
        kind = "interface" if "interface" in text else "type"
        result.definitions.append((m.group(1), lineno, kind))

    for m in _JS_IMPORT_FROM_RE.finditer(source):
        module = m.group(1) or m.group(2)
        result.imports.append(module)

    return result


# ── Go parser (regex) ────────────────────────────────────────────────────

_GO_FUNC_RE = re.compile(
    r"^func\s+(?:\(\s*\w+\s+\*?\w+\s*\)\s+)?(\w+)\s*\(",
    re.MULTILINE,
)
_GO_TYPE_RE = re.compile(
    r"^type\s+(\w+)\s+(struct|interface)\b",
    re.MULTILINE,
)
_GO_IMPORT_SINGLE_RE = re.compile(
    r"""^\s*import\s+"([^"]+)"\s*$""",
    re.MULTILINE,
)
_GO_IMPORT_BLOCK_RE = re.compile(
    r"import\s*\((.*?)\)",
    re.DOTALL,
)
_GO_IMPORT_LINE_RE = re.compile(r'"([^"]+)"')


def _parse_go(source: str, file_id: str) -> ParseResult:
    result = ParseResult()

    for m in _GO_FUNC_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        result.definitions.append((m.group(1), lineno, "function"))

    for m in _GO_TYPE_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        result.definitions.append((m.group(1), lineno, m.group(2)))

    for m in _GO_IMPORT_SINGLE_RE.finditer(source):
        result.imports.append(m.group(1))

    for m in _GO_IMPORT_BLOCK_RE.finditer(source):
        for line_m in _GO_IMPORT_LINE_RE.finditer(m.group(1)):
            result.imports.append(line_m.group(1))

    return result


# ── Rust parser (regex) ──────────────────────────────────────────────────

_RUST_FN_RE = re.compile(
    r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)",
    re.MULTILINE,
)
_RUST_STRUCT_RE = re.compile(
    r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?struct\s+(\w+)",
    re.MULTILINE,
)
_RUST_TRAIT_RE = re.compile(
    r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?trait\s+(\w+)",
    re.MULTILINE,
)
_RUST_IMPL_RE = re.compile(
    r"^\s*impl(?:\s*<[^>]*>)?\s+(\w+)",
    re.MULTILINE,
)
_RUST_ENUM_RE = re.compile(
    r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?enum\s+(\w+)",
    re.MULTILINE,
)
_RUST_USE_RE = re.compile(
    r"^\s*(?:pub\s+)?use\s+([\w:]+)",
    re.MULTILINE,
)


def _parse_rust(source: str, file_id: str) -> ParseResult:
    result = ParseResult()

    for m in _RUST_FN_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        kind = "async_function" if "async" in m.group(0) else "function"
        result.definitions.append((m.group(1), lineno, kind))

    for m in _RUST_STRUCT_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        result.definitions.append((m.group(1), lineno, "struct"))

    for m in _RUST_TRAIT_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        result.definitions.append((m.group(1), lineno, "trait"))

    for m in _RUST_IMPL_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        result.definitions.append((m.group(1), lineno, "impl"))

    for m in _RUST_ENUM_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        result.definitions.append((m.group(1), lineno, "enum"))

    for m in _RUST_USE_RE.finditer(source):
        # "std::collections::HashMap" -> "std::collections"
        path = m.group(1)
        parts = path.split("::")
        if len(parts) > 1:
            result.imports.append("::".join(parts[:-1]))
        else:
            result.imports.append(path)

    return result


# ── C/C++ parser (regex) ────────────────────────────────────────────────

_C_FUNC_RE = re.compile(
    r"^(?!.*\b(?:if|for|while|switch|return|else|catch|throw)\b)"
    r"[\w\s\*&:<>,]+?\s+(\w+)\s*\([^;]*\)\s*\{",
    re.MULTILINE,
)
_C_CLASS_RE = re.compile(
    r"^\s*(?:class|struct)\s+(\w+)",
    re.MULTILINE,
)
_C_NAMESPACE_RE = re.compile(
    r"^\s*namespace\s+(\w+)",
    re.MULTILINE,
)
_C_INCLUDE_RE = re.compile(
    r"""^\s*#include\s+[<"]([^>"]+)[>"]""",
    re.MULTILINE,
)


def _parse_c_cpp(source: str, file_id: str) -> ParseResult:
    result = ParseResult()

    for m in _C_FUNC_RE.finditer(source):
        name = m.group(1)
        # Skip common false positives
        if name in ("if", "for", "while", "switch", "return", "else",
                     "catch", "throw", "main", "class", "struct",
                     "namespace", "typedef", "enum", "union"):
            continue
        lineno = source[: m.start()].count("\n") + 1
        result.definitions.append((name, lineno, "function"))

    for m in _C_CLASS_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        text = m.group(0)
        kind = "class" if "class" in text else "struct"
        result.definitions.append((m.group(1), lineno, kind))

    for m in _C_NAMESPACE_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        result.definitions.append((m.group(1), lineno, "namespace"))

    for m in _C_INCLUDE_RE.finditer(source):
        result.imports.append(m.group(1))

    return result


# ── Ruby parser (regex) ─────────────────────────────────────────────────

_RUBY_DEF_RE = re.compile(
    r"^\s*def\s+(self\.)?(\w+[!?=]?)",
    re.MULTILINE,
)
_RUBY_CLASS_RE = re.compile(
    r"^\s*class\s+(\w+)",
    re.MULTILINE,
)
_RUBY_MODULE_RE = re.compile(
    r"^\s*module\s+(\w+)",
    re.MULTILINE,
)
_RUBY_REQUIRE_RE = re.compile(
    r"""^\s*require(?:_relative)?\s+['"]([^'"]+)['"]""",
    re.MULTILINE,
)


def _parse_ruby(source: str, file_id: str) -> ParseResult:
    result = ParseResult()

    for m in _RUBY_DEF_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        name = m.group(2)
        kind = "class_method" if m.group(1) else "method"
        result.definitions.append((name, lineno, kind))

    for m in _RUBY_CLASS_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        result.definitions.append((m.group(1), lineno, "class"))

    for m in _RUBY_MODULE_RE.finditer(source):
        lineno = source[: m.start()].count("\n") + 1
        result.definitions.append((m.group(1), lineno, "module"))

    for m in _RUBY_REQUIRE_RE.finditer(source):
        result.imports.append(m.group(1))

    return result


# ── Language routing ─────────────────────────────────────────────────────

_LANGUAGE_MAP: dict[str, Callable[[str, str], ParseResult]] = {
    ".py": _parse_python,
    ".java": _parse_java,
    ".js": _parse_javascript,
    ".jsx": _parse_javascript,
    ".ts": _parse_javascript,
    ".tsx": _parse_javascript,
    ".go": _parse_go,
    ".rs": _parse_rust,
    ".c": _parse_c_cpp,
    ".h": _parse_c_cpp,
    ".cpp": _parse_c_cpp,
    ".hpp": _parse_c_cpp,
    ".cc": _parse_c_cpp,
    ".rb": _parse_ruby,
}

_ALL_EXTENSIONS = sorted(_LANGUAGE_MAP.keys())

# Known extensions for stripping in import resolution.
_KNOWN_EXTENSIONS = set(_LANGUAGE_MAP.keys())


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class CrawlConfig:
    """Controls which files/directories the crawler visits."""

    extensions: list[str] = field(default_factory=lambda: list(_ALL_EXTENSIONS))
    ignore_patterns: list[str] = field(
        default_factory=lambda: [
            "__pycache__",
            ".git",
            ".venv",
            "node_modules",
            ".pytest_cache",
            "target",
            "build",
            "dist",
            "vendor",
        ]
    )


# ── Public API ───────────────────────────────────────────────────────────

def crawl_project(
    root_path: str | Path,
    config: CrawlConfig | None = None,
) -> EntityGraph:
    """Walk *root_path*, parse every matching source file, and return a
    populated :class:`EntityGraph` containing FILE, FUNCTION, and CONCEPT
    nodes together with PART_OF, IMPORTS, and CALLS edges.

    Supports Python, Java, JS/TS, Go, Rust, C/C++, and Ruby.
    """
    config = config or CrawlConfig()
    root = Path(root_path).resolve()
    graph = EntityGraph()

    # Collect all matching source files.
    source_files = _collect_files(root, config)
    log.info("Crawling %s: found %d source files", root, len(source_files))

    # First pass: create FILE and FUNCTION nodes + PART_OF edges, and
    # record raw import/call data for the second pass.
    file_imports: dict[str, list[str]] = {}       # rel_path -> [module_name, ...]
    func_calls: dict[str, list[str]] = {}         # func_node_id -> [called_name, ...]

    for src in source_files:
        rel = src.relative_to(root)
        rel_id = rel.as_posix()                   # forward-slash node_id

        # FILE node -------------------------------------------------------
        try:
            source_text = src.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            log.warning("Could not read %s: %s", src, exc)
            continue

        line_count = source_text.count("\n") + (1 if source_text else 0)
        graph.add_node(
            node_id=rel_id,
            node_type=NodeType.FILE,
            label=src.name,
            metadata={"path": str(rel), "lines": line_count},
        )

        # Dispatch to language-specific parser ----------------------------
        ext = src.suffix.lower()
        parser = _LANGUAGE_MAP.get(ext)
        if parser is None:
            log.debug("No parser for extension %s (%s), skipping parse", ext, src)
            continue

        parse_result = parser(source_text, rel_id)

        # Create FUNCTION nodes + PART_OF edges from definitions ----------
        for name, lineno, kind in parse_result.definitions:
            node_id = f"{rel_id}::{name}"
            graph.add_node(
                node_id=node_id,
                node_type=NodeType.FUNCTION,
                label=name,
                metadata={"lineno": lineno, "type": kind},
            )
            graph.add_edge(node_id, rel_id, EdgeType.PART_OF)

        # Record imports for second pass ----------------------------------
        file_imports[rel_id] = parse_result.imports

        # Record calls for second pass ------------------------------------
        for caller_id, called_names in parse_result.calls:
            func_calls[caller_id] = called_names

    # Second pass: resolve imports and calls against known nodes.
    _resolve_imports(file_imports, graph, root)
    _resolve_calls(func_calls, graph)

    log.info("Crawl complete: %d nodes, %d edges", graph.node_count, graph.edge_count)
    return graph


# ── File collection ──────────────────────────────────────────────────────

def _collect_files(root: Path, config: CrawlConfig) -> list[Path]:
    """Recursively collect files under *root* that match *config*."""
    results: list[Path] = []
    for item in sorted(root.rglob("*")):
        # Skip anything whose path components match an ignore pattern.
        if _should_ignore(item, root, config.ignore_patterns):
            continue
        if item.is_file() and item.suffix in config.extensions:
            results.append(item)
    return results


def _should_ignore(path: Path, root: Path, patterns: list[str]) -> bool:
    """Return True if any component of *path* (relative to *root*) matches
    one of the ignore *patterns* via ``fnmatch``."""
    rel = path.relative_to(root)
    for part in rel.parts:
        for pat in patterns:
            if fnmatch(part, pat):
                return True
    return False


# ── Resolution passes ────────────────────────────────────────────────────

def _resolve_imports(
    file_imports: dict[str, list[str]],
    graph: EntityGraph,
    root: Path,
) -> None:
    """Create IMPORTS edges.  If the imported module resolves to a project
    file, point to that file's node; otherwise create a CONCEPT node for
    the external module."""
    # Build a lookup: module dotted path -> file node_id for all known files.
    file_node_ids = set(graph.get_nodes_by_type(NodeType.FILE))
    module_to_file: dict[str, str] = {}
    for fid in file_node_ids:
        # Strip any known extension suffix: "pkg/sub/mod.py" -> "pkg.sub.mod"
        fid_stripped = fid
        for ext in _KNOWN_EXTENSIONS:
            if fid_stripped.endswith(ext):
                fid_stripped = fid_stripped[: -len(ext)]
                break
        dotted = fid_stripped.replace("/", ".")
        module_to_file[dotted] = fid
        # Also map just the filename stem for single-file modules.
        stem = Path(fid).stem
        if stem not in module_to_file:
            module_to_file[stem] = fid
        # Also map the raw file path (without extension) for JS-style imports
        # like "./utils" -> "utils.js"
        module_to_file[fid_stripped] = fid

    for file_id, modules in file_imports.items():
        for mod in modules:
            target_id: str
            # Normalize JS-style relative imports: "./utils" -> "utils"
            normalized = mod.lstrip("./")
            if normalized in module_to_file:
                target_id = module_to_file[normalized]
            elif mod in module_to_file:
                target_id = module_to_file[mod]
            else:
                # Try dotted form for Go/Java/Rust paths
                dotted_mod = mod.replace("/", ".").replace("::", ".")
                if dotted_mod in module_to_file:
                    target_id = module_to_file[dotted_mod]
                else:
                    # External module -> CONCEPT node
                    target_id = mod
                    if not graph.has_node(target_id):
                        graph.add_node(
                            target_id,
                            NodeType.CONCEPT,
                            label=mod,
                            metadata={"kind": "external_module"},
                        )
            graph.add_edge(file_id, target_id, EdgeType.IMPORTS)


def _resolve_calls(
    func_calls: dict[str, list[str]],
    graph: EntityGraph,
) -> None:
    """Create CALLS edges for calls that match a known FUNCTION node by
    name suffix (i.e. the label)."""
    # Build name -> [node_id, ...] index for all FUNCTION nodes.
    name_index: dict[str, list[str]] = {}
    for nid in graph.get_nodes_by_type(NodeType.FUNCTION):
        node_data = graph.get_node(nid)
        if node_data:
            label = node_data["label"]
            name_index.setdefault(label, []).append(nid)

    for caller_id, called_names in func_calls.items():
        seen: set[str] = set()
        for name in called_names:
            if name in name_index:
                for target_id in name_index[name]:
                    # Avoid self-calls and duplicate edges in the same caller.
                    if target_id == caller_id:
                        continue
                    pair = (caller_id, target_id)
                    if pair not in seen:
                        seen.add(pair)
                        graph.add_edge(caller_id, target_id, EdgeType.CALLS)
