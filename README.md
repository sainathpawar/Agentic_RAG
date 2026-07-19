# RAG & Agentic RAG

Two runnable scripts that map **1:1 to the tutorial diagram**, so you can point at a
box on screen and then at the matching line of code.

## Theory slides (explain first)
Ready-made diagrams for the concept part of the video:
- `basic_rag_diagram.png` — the 7-step RAG flow
- `agentic_rag_diagram.png` — the 11-step Agentic RAG flow (decisions + self-correcting loop)

Regenerate/tweak them with: `python make_diagrams.py`

## The free stack
| Piece | Tool | Why |
|-------|------|-----|
| LLM | **Groq API** (Llama 3.3 70B) | Free tier, extremely fast |
| Embeddings | **HuggingFace MiniLM** | Runs LOCALLY, no API key, free |
| Vector DB | **FAISS** | Local, no server |
| Orchestration | **LangChain** + **LangGraph** | Framework for RAG + agent loops |
| Internet source | **DuckDuckGo** | Free web search, no key |

## Setup (once)
```bash
pip install -r requirements.txt
cp .env.example .env          # then paste your free key from console.groq.com/keys
```

## Run
```bash
python 01_basic_rag.py        # the simple RAG pipeline
python 02_agentic_rag.py      # the agent that decides + loops
python export_graph.py        # save the LangGraph as a picture (.mmd / .png / ASCII)
```

Prefer notebooks for teaching? Open **`RAG_and_Agentic_RAG.ipynb`** — same content,
cell by cell, runs locally or in Google Colab (it has a `%pip install` cell and a
Colab key-entry snippet).

## Show the graph on screen
`export_graph.py` writes:
- `agentic_rag_graph.mmd` — Mermaid text; paste into <https://mermaid.live> (always works)
- `agentic_rag_graph.png` — rendered image (needs internet or local graphviz)
- an ASCII sketch printed to the terminal (great for a quick screen-share)

Inside the notebook, the "Visualise the graph" cell renders the same picture inline
with `agent.get_graph().draw_mermaid_png()`.

---

## Part 1 — Basic RAG  (`01_basic_rag.py`)
Follows the **top** diagram, 7 steps:

| # | Diagram box | Code |
|---|-------------|------|
| 1 | Query | `input("You: ")` |
| 2 | Encode | `HuggingFaceEmbeddings(...)` |
| 3 | Vector Database (Index) | `FAISS.from_documents(...)` |
| 4 | Similarity Search | `vector_store.as_retriever(...)` |
| 5 | Retrieved Context + Query | the `ChatPromptTemplate` |
| 6 | LLM | `ChatGroq(...)` |
| 7 | Response | `StrOutputParser()` output |

> 🎬 **Teaching note:** the diagram labels box 2 "LLM", but the encoder is really a
> dedicated **embeddings model** — a good point to clarify on camera.

---

## Part 2 — Agentic RAG  (`02_agentic_rag.py`)
Follows the **bottom** diagram, 11 steps. The difference from basic RAG is that the
LLM now **makes decisions** (diamonds) and can **loop**. Built with LangGraph.

| # | Diagram box | Code |
|---|-------------|------|
| 1 | Query | `agent.invoke({"question": ...})` |
| 2 | Rewrite Query | `rewrite_query` node |
| 3 | Updated Query | `state["updated_query"]` |
| 4 | Need More Details? (YES/NO) | `decide_to_retrieve` |
| 5 | Which Source? | `route_to_source` |
| 6 | Sources (Vector DB / Internet) | `retrieve` node |
| 7 | Retrieved Context + Updated Query | `state["context"]` |
| 8 | LLM | `generate` node |
| 9 | Response | `state["generation"]` |
| 10 | Is the answer relevant? (YES/NO) | `grade_answer` |
| 11 | Final Response | returned when grade = YES |

**The two agentic superpowers to highlight on camera:**
1. **Decision (step 4/5):** the agent can skip retrieval for simple questions, or
   choose the *right* source (private notes vs the open web).
2. **Self-correction loop (step 10):** if the answer isn't relevant, it goes back to
   step 2 and tries again — with a retry cap so it never loops forever.

### Demo questions to show the branches
- `What is a transformer?` → routes to **Vector DB** (it's in our notes).
- `Who won the 2022 FIFA World Cup?` → routes to **Internet**.
- `What is 15 times 12?` → **NO** retrieval, LLM answers directly.
