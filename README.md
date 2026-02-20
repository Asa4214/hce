# Holographic Context Engine (HCE)

A smart memory system for AI assistants. Instead of dumping entire conversation history into the context window, HCE retrieves only the most relevant memories — like how your brain recalls related experiences, not your entire life story.

```
User Query
    |
    v
+---------------------------+
|     HCE Middleware         |
|                           |
|  +---------+  +--------+  |
|  | Entity  |  |Semantic|  |
|  | Graph   |  | Tree   |  |
|  +---------+  +--------+  |
|        +--------+         |
|        | Focus  |         |
|        | Buffer |         |
|        +--------+         |
|           |               |
|    Context Budgeting      |
|    (pick best memories)   |
+---------------------------+
    |
    v
[Context Block + Query] --> LLM --> Response
                                      |
                              Store back into HCE
```

## How It Works

HCE uses **three parallel memory structures**, inspired by how human memory works:

| Structure | Human Analogy | What It Stores | How It Retrieves |
|-----------|--------------|----------------|------------------|
| **Entity Graph** | Associative memory ("reminds me of...") | Concepts, files, people, and their relationships | Spreading Activation — energy propagates through connections |
| **Semantic Tree** | Episodic memory (past experiences) | Every conversation turn, organized hierarchically | Hierarchical Relevance Search — drill into relevant branches |
| **Focus Buffer** | Short-term memory (last few minutes) | Last N conversation turns | Recency — most recent gets priority |

A **Context Budgeting** algorithm (greedy knapsack) then picks the best memories that fit within the LLM's token budget, scored by `Utility / Token_Cost`.

## Project Structure

```
hce-project/
├── hce_core.py            # EntityGraph (networkx) + Spreading Activation
├── semantic_tree.py        # SemanticTree + Hierarchical Relevance Search
├── entity_extractor.py     # Regex/heuristic NER for text -> entities
├── project_crawler.py      # Multi-language codebase indexer (Python/Java/JS/TS/Go/Rust/C/C++/Ruby)
├── hce_pipeline.py         # Pipeline orchestrator + Focus Buffer + Context Budgeting
├── hce_mcp_server.py       # MCP server for Claude Code integration
├── test_hce_core.py        # Tests for EntityGraph + Spreading Activation
├── test_semantic_tree.py   # Tests for SemanticTree + HRS
├── test_entity_extractor.py# Tests for entity extraction
├── test_project_crawler.py # Tests for project crawler
├── test_hce_pipeline.py    # Tests for pipeline + buffer + budgeting
├── architecture_plan.md    # Original design document
├── pyproject.toml          # Dependencies and project metadata
├── CLAUDE.md               # Instructions for Claude Code
└── .mcp.json               # MCP server registration
```

## Quick Start

```bash
# Clone and set up
git clone <repo-url>
cd hce-project
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Use with Claude Code (MCP integration)
# Just open the project in Claude Code — the MCP server starts automatically
```

## Usage

### As a Python library

```python
from hce_pipeline import HCEPipeline

# Create a pipeline
pipeline = HCEPipeline(context_budget=4000)

# Store a conversation turn
pipeline.update(
    user_query="What is spreading activation?",
    ai_response="Spreading activation is an algorithm that propagates energy through a graph..."
)

# Later, retrieve relevant context for a new query
context = pipeline.retrieve_context("How does the entity graph find related concepts?")
print(context)
# >> ## Related Knowledge
# >> - [concept] spreading_activation (score: 0.85)
# >> - [concept] entity_graph (score: 0.72)
# >> ...

# Or wrap an LLM chat function
def my_chat(prompt: str) -> str:
    # call your LLM here
    return llm.generate(prompt)

smart_chat = pipeline.wrap_chat(my_chat)
response = smart_chat("How does the entity graph find related concepts?")
# HCE automatically enriches the prompt with context and stores the result
```

### Index a codebase

```python
from project_crawler import crawl_project

# Supports Python, Java, JS/TS, Go, Rust, C/C++, Ruby
graph = crawl_project("/path/to/your/project")
print(f"Found {graph.node_count} entities and {graph.edge_count} relationships")
```

### Persistence

```python
# Save state
pipeline.save("~/.hce_state")

# Load later
pipeline = HCEPipeline.load("~/.hce_state")
```

### With Claude Code & GitHub Copilot CLI (MCP)

HCE integrates with any tool that supports the [Model Context Protocol](https://modelcontextprotocol.io), including **Claude Code** and **GitHub Copilot CLI**. It provides persistent memory across sessions — automatically storing important exchanges and retrieving relevant context.

#### Claude Code Setup

Add this to your project's `.mcp.json` (adjust paths to your environment):

```json
{
  "mcpServers": {
    "hce": {
      "command": "/path/to/your/.venv/bin/python",
      "args": ["/path/to/hce_mcp_server.py"]
    }
  }
}
```

Then add the auto-behavior rules from `CLAUDE.md` to your project so Claude Code calls HCE tools automatically.

#### GitHub Copilot CLI Setup

Add HCE to your Copilot CLI MCP config at `~/.copilot/mcp-config.json`:

```json
{
  "mcpServers": {
    "hce": {
      "type": "local",
      "command": "/path/to/your/.venv/bin/python",
      "args": ["/path/to/hce_mcp_server.py"]
    }
  }
}
```

> **Note:** Claude Code reads `CLAUDE.md` for auto-behavior rules (when to store/retrieve automatically). Copilot CLI does not — you'll need to explicitly ask it to use HCE tools, or configure similar rules in Copilot's system prompt.

#### Other MCP-Compatible Tools

HCE works with any MCP client, including **VS Code** (Copilot Chat), **Cursor**, **Windsurf**, **Zed**, and **Claude Desktop**. Check each tool's docs for how to register a custom MCP server.

#### MCP Tools

| Tool | Purpose | When it's called |
|------|---------|-----------------|
| `hce_status` | Show memory statistics (nodes, edges, interactions, buffer) | On session start |
| `hce_retrieve_context` | Search all 3 structures and return curated context within token budget | Before answering codebase/architecture/history questions |
| `hce_store_interaction` | Save a conversation turn into graph, tree, and buffer | After important exchanges (design decisions, bug fixes, features) |
| `hce_crawl_project` | Index a codebase (Python, Java, JS/TS, Go, Rust, C/C++, Ruby) into the entity graph | When graph is empty or user requests re-indexing |
| `hce_search_graph` | Search the entity graph via spreading activation | When looking for related code entities or concepts |
| `hce_clear_memory` | Reset all HCE state (graph, tree, buffer) | Only when explicitly requested |

#### Example: What happens in a session

```
Session Start:
  Claude calls hce_status → sees 313 nodes, 7 stored interactions
  Graph already populated → no need to crawl

User: "How does the authentication system work?"
  Claude calls hce_retrieve_context("authentication system")
  → HCE returns: 2 graph entities (login.py, validate_token),
    1 past conversation about auth design, last 2 recent turns
  → All packed within 4,000 token budget
  Claude answers using that context

User: "Let's switch from JWT to session cookies"
  Claude implements the change, then calls hce_store_interaction(...)
  → Decision stored in all 3 structures for future sessions

Next Session (days later):
  User: "Why did we switch away from JWT?"
  Claude calls hce_retrieve_context → finds the stored decision
  Claude: "We switched to session cookies because..."
```

#### All data is local

HCE stores everything at `~/.hce_state/` on your machine. No cloud, no external API calls. See the [FAQ](FAQ.md) for more details.

## FAQ

**"I installed HCE but my context keeps growing — is it broken?"** No! HCE doesn't shrink your current conversation. It provides cross-session memory and smart retrieval for LLM apps. See the full [FAQ](FAQ.md) for this and other common questions.

## Architecture Deep Dive

### Entity Graph (`hce_core.py`)

A typed property graph backed by `networkx.MultiDiGraph`.

- **Node types:** File, Function, Concept, Person, Event
- **Edge types:** Imports, Calls, Relates_To, Part_Of
- **Retrieval:** Spreading Activation — seed nodes get energy, which propagates to neighbors with a decay factor

```
[login.py] --Imports--> [user_model.py] --Part_Of--> [auth/]
     |                        |
  Calls                   Calls
     v                        v
[validate()]           [get_user()]
```

### Semantic Tree (`semantic_tree.py`)

A Merkle/Aggregation tree where leaves are raw conversation turns and internal nodes are summaries.

```
        [Root: summary of everything]
       /                            \
  [Summary: turns 1-4]        [Summary: turns 5-8]
   /    |    |    \             /    |    |    \
 T1    T2   T3   T4          T5   T6   T7   T8   <-- leaves (raw turns)
```

**Hierarchical Relevance Search (HRS):** Start at the root, compute similarity, recurse into promising branches, prune low-scoring ones.

### Context Budgeting (`hce_pipeline.py`)

A greedy knapsack algorithm that selects the best memories within a token budget:

1. Collect candidates from all three structures
2. Score each by `Utility / Token_Cost` (efficiency ratio)
3. Greedily pack the highest-efficiency candidates until budget is full

## Tech Stack

- **Python 3.10+**
- **networkx** — Graph data structure
- **mcp** — Model Context Protocol for Claude Code integration
- **pytest** — Testing

## Current Status

All 4 implementation phases are complete:

- [x] Phase 1: EntityGraph + Spreading Activation
- [x] Phase 2: SemanticTree + Hierarchical Relevance Search
- [x] Phase 3: Project Crawler + Entity Extractor
- [x] Phase 4: Pipeline Middleware + MCP Server

**194 tests passing.** See the [architecture plan](architecture_plan.md) for the full design rationale.

## Scalability

HCE works well for single-developer local use (up to ~10K graph nodes, hundreds of interactions). For scaling beyond that, see the [Scalability Guide](SCALABILITY.md) — it covers current capacity, known bottlenecks, and the recommended upgrade path (SQLite backend, semantic embeddings, incremental tree insertion).

## Known Limitations

- **Vectorizer:** Uses feature-hashing (bag-of-words), not semantic embeddings. "car" and "automobile" won't match.
- **Summarizer:** Extractive (first sentences), not abstractive. No LLM summarization yet.
- **NER:** Regex/heuristic-based, not ML. Misses many entities in natural text.
- **Non-Python parsers:** Java, JS/TS, Go, Rust, C/C++, and Ruby use regex-based parsing (no semantic understanding, may miss complex patterns). Python uses full AST parsing.
- **Platform:** File locking uses POSIX `fcntl.flock()` — not available on Windows without adaptation.

For a full analysis of scaling limits and fixes, see [SCALABILITY.md](SCALABILITY.md).

## License

MIT
