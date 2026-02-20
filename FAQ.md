# Frequently Asked Questions

## "I installed HCE but my Claude Code context bar keeps growing. Is it broken?"

**No — this is completely normal.** This is the #1 most common misunderstanding, so let's clear it up.

There are **two different "contexts"** at play, and HCE only controls one of them:

```
Context #1 (NOT controlled by HCE)          Context #2 (controlled by HCE)
┌──────────────────────────────────┐        ┌──────────────────────────────────┐
│  Claude Code Conversation        │        │  HCE Memory Retrieval            │
│                                  │        │                                  │
│  You: "Fix the login bug"        │        │  What HCE injects into a prompt: │
│  Claude: "I see the issue..."    │        │                                  │
│  You: "Now add tests"            │        │  - 3 relevant graph entities     │
│  Claude: "Here are the tests..." │        │  - 1 past conversation match     │
│  ...keeps growing...             │        │  - last 2 recent turns           │
│                                  │        │                                  │
│  This is Claude Code's own       │        │  Packed into a fixed budget      │
│  conversation history. HCE       │        │  (default 4,000 tokens).         │
│  can't shrink this.              │        │  Always stays within budget.     │
└──────────────────────────────────┘        └──────────────────────────────────┘
```

**Claude Code manages its own conversation context.** Every message you send and every response you get adds to it. HCE sits alongside as a tool — it can't remove or replace Claude Code's native history.

### So what DOES HCE do?

HCE solves a different problem: **cross-session memory and smart retrieval.**

| Scenario | Without HCE | With HCE |
|----------|-------------|----------|
| New session after closing Claude Code | AI remembers nothing | AI retrieves past decisions, architecture, bug fixes |
| "What did we decide about the auth system?" | "I don't have context on that" | Finds the stored conversation automatically |
| Large codebase questions | You re-explain everything each time | Entity Graph already maps the code relationships |
| Building your own LLM app | You dump full chat history into the prompt (grows forever) | You call `retrieve_context()` and get a fixed-budget curated result |

### Where HCE really shines

HCE is designed as **middleware for LLM applications**. If you're building a chatbot, copilot, or agent, here's the difference:

```python
# WITHOUT HCE — context grows linearly with conversation length
prompt = system_message + entire_conversation_history + user_query
# 100 turns = 100 turns of tokens crammed in

# WITH HCE — context stays within budget
context = pipeline.retrieve_context(user_query)  # Always <= 4,000 tokens
prompt = system_message + context + user_query
# 100 turns or 10,000 turns — same budget, just smarter selection
```

---

## "How is HCE different from just using RAG?"

RAG (Retrieval-Augmented Generation) typically does one thing: vector similarity search over documents.

HCE uses **three parallel retrieval strategies** that each catch different kinds of relevance:

| | RAG | HCE |
|---|---|---|
| **Structure** | Flat vector store | Graph + Tree + Buffer (3 structures) |
| **Relationships** | None — each chunk is independent | Entity Graph tracks how concepts connect |
| **Recency** | No awareness of time | Focus Buffer prioritizes recent turns |
| **Hierarchy** | Flat chunks | Semantic Tree drills into relevant branches, skips the rest |
| **Budget control** | Return top-K results (may overshoot or undershoot) | Greedy knapsack packs best results into exact token budget |

Think of it this way: RAG is like searching Google. HCE is like having a research assistant who knows the connections between topics, remembers your recent conversation, and has a summary of everything you've ever discussed — then picks the best combination of all three to brief you.

---

## "Does HCE store my conversations on a remote server?"

**No.** Everything is stored locally at `~/.hce_state/` on your machine. HCE is fully local:

- `entity_graph.json` — your Entity Graph
- `semantic_tree.json` — your Semantic Tree
- `pipeline_state.json` — Focus Buffer and pipeline config

No data leaves your machine. No cloud. No API calls for storage.

---

## "Can I use HCE in my own app (not just Claude Code)?"

Yes! HCE is a Python library first, and an MCP server second. You can use it directly:

```python
from hce_pipeline import HCEPipeline

pipeline = HCEPipeline(context_budget=4000)

# Store conversations
pipeline.update(
    user_query="How do I deploy to production?",
    ai_response="You need to run the deploy script..."
)

# Later, retrieve relevant context for any new query
context = pipeline.retrieve_context("What are the deployment steps?")
# Returns the most relevant memories within 4,000 tokens
```

You can wrap any LLM chat function:

```python
smart_chat = pipeline.wrap_chat(my_llm_function)
response = smart_chat("Remind me about the deployment steps")
# HCE enriches the prompt automatically and stores the result
```

---

## "What's the token budget and can I change it?"

The default budget is **4,000 tokens**. You can change it when creating a pipeline:

```python
pipeline = HCEPipeline(context_budget=8000)  # larger budget
pipeline = HCEPipeline(context_budget=2000)  # smaller budget
```

The budget controls the **maximum** amount of context HCE will inject. It uses a greedy knapsack algorithm to pack the highest `Utility / Token_Cost` items first, so even a small budget gets the most relevant information.

---

## "How does HCE choose which memories to include?"

Three-step process:

1. **Collect candidates** from all three structures:
   - Entity Graph: concepts/entities related to your query (via spreading activation)
   - Semantic Tree: past conversations that match your query (via hierarchical search)
   - Focus Buffer: your most recent turns (by recency)

2. **Score each candidate** by its efficiency: `Utility / Token_Cost`
   - A highly relevant 50-token memory scores better than a somewhat relevant 500-token one

3. **Greedy packing** — add candidates highest-efficiency-first until the budget is full

This means HCE always fits within budget, and the most information-dense memories get priority.

---

## "Can HCE index non-Python codebases?"

Yes. HCE supports multiple languages:

| Language | Parser Type |
|----------|-------------|
| Python | Full AST parsing (most accurate) |
| Java | Regex-based |
| JavaScript / TypeScript | Regex-based |
| Go | Regex-based |
| Rust | Regex-based |
| C / C++ | Regex-based |
| Ruby | Regex-based |

Python gets the deepest understanding (full abstract syntax tree parsing). Other languages use regex-based parsers that catch function definitions, imports, and calls but may miss complex patterns.

---

## "Is HCE safe with multiple Claude Code windows open?"

Yes. HCE uses **POSIX file locking** (`fcntl.flock()`) to protect the state directory:

- **Reading** acquires a shared lock (multiple readers allowed)
- **Writing** acquires an exclusive lock (blocks all other readers/writers)
- Locks have a 10-second timeout to prevent deadlocks

This means multiple MCP server instances can safely read/write to `~/.hce_state/` without corrupting data.

**Note:** File locking uses POSIX `fcntl.flock()` and is not available on Windows without adaptation.
