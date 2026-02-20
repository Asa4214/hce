# System Architecture: The "Holographic Context" Engine (HCE)

This document outlines a Data Structure and Algorithm (DSA) centric approach to solving the AI context window limitation. Instead of just "stuffing" more tokens, we treat Context as a **Dynamic Retrieval Problem**, creating a "Cognitive Operating System" for AI agents.

## 1. The Core Philosophy
We move away from a linear "Conversation History" (List) and towards a **Multi-Model Memory System**. The agent doesn't just "read" history; it queries its own internal database to construct a "Working Memory" for every single prompt.

## 2. Data Structures (The "Storage Layer")

To handle infinite context, we use three distinct data structures working in parallel:

### A. The Semantic Tree (Long-Term Episodic Memory)
*   **Structure:** **Merkle Tree / Aggregation Tree**.
    *   **Leaves:** Raw interaction turns (User prompt + AI response).
    *   **Nodes:** Vectorized summaries of their children.
*   **Why:** Allows the agent to retrieve high-level summaries of old events while keeping exact details for relevant topics.

### B. The Entity Graph (Associative Memory)
*   **Structure:** **Property Graph** (Nodes & Edges).
    *   **Nodes:** `File`, `Function`, `Concept`, `Person`, `Event`.
    *   **Edges:** `Imports`, `Calls`, `Relates_To`, `Part_Of`.
*   **Why:** Enables "Associative Recall". It pulls in logically related nodes (e.g., `user_model.py` when discussing `login.py`) even without semantic similarity.

### C. The Sliding Focus Buffer (Short-Term Memory)
*   **Structure:** **Circular Buffer (Queue)**.
*   **Why:** Maintains immediate continuity for the last N interactions.

---

## 3. Algorithms (The "Retrieval Layer")

### Algorithm 1: Hierarchical Relevance Search (HRS)
*   **Logic:** Recursive vector similarity starting from the root of the Semantic Tree. Prunes irrelevant branches early to save compute.

### Algorithm 2: Spreading Activation
*   **Logic:** "Energizes" nodes found in the query and propagates that energy to neighbors using a decay factor. Selects top-N most "active" nodes.

### Algorithm 3: Context Budgeting (The "Knapsack Problem")
*   **Logic:** Scores all candidates from the Tree and Graph. Greedily fills the context window based on `(Utility / Token_Cost)`.

---

## 4. Integration Strategy: The "Middleware Wrapper"

HCE acts as a **Smart Memory Management Unit (MMU)** that sits between the User and the LLM (Gemini/Claude).

1.  **Intercept:** User query is intercepted by the HCE Wrapper.
2.  **Retrieve:** HCE queries the Graph and Tree to find relevant context.
3.  **Synthesize:** HCE packs the retrieved data into a "Context Block."
4.  **Inference:** The LLM receives the prompt: `[HCE Context Block] + [Focus Buffer] + [User Query]`.
5.  **Update:** The response is stored back into the HCE structures.

---

## 5. Beyond Code: General Conversation Handling

The HCE is **not** limited to codebases. It treats "Normal Conversations" as a Knowledge Graph construction problem.

*   **Entity Extraction:** Every message is processed via Named Entity Recognition (NER) to find People, Places, and Topics.
*   **Semantic Edges:** If a user says "I'm planning a trip to Japan," the graph creates a node for `Japan` and links it to `Trip Planning`.
*   **Relational Memory:** In a normal conversation, the "Edges" are semantic relations (e.g., "Alice is a doctor") rather than code imports.
*   **The Result:** If you talk about "Hospitals" six months later, the **Spreading Activation** algorithm will travel from `Hospital` -> `Doctor` -> `Alice`, bringing your friend Alice's details back into context automatically.

---

## 6. Market Analysis & Competitive Positioning (2025-2026)

### Comparison Table
| Feature | Basic Vector RAG | Microsoft GraphRAG | MemGPT | **HCE (Our Solution)** |
| :--- | :--- | :--- | :--- | :--- |
| **Data Structure** | Flat Vector List | Static Knowledge Graph | Virtual Paging | **Graph + Tree + Queue** |
| **Logic** | Semantic Similarity | Community Detection | OS Paging/Function Calls | **Spreading Activation** |
| **Recall Type** | Textual Match | Global Summarization | Temporal Memory | **Associative & Adaptive** |
| **Resolution** | Fixed Chunks | Static Communities | Block-based | **Multi-resolution (Zoom)** |

---

## 7. Implementation Roadmap
1.  **Phase 1:** Implement Python-based `EntityGraph` using `networkx` and the **Spreading Activation** logic.
2.  **Phase 2:** Build the `SemanticTree` with recursive summarization.
3.  **Phase 3:** Create the **Project Crawler** (for code) and **Entity Extractor** (for chat).
4.  **Phase 4:** Wrap the Agent's `chat` function with the HCE Pipeline.
