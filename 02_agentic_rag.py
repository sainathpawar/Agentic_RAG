"""
================================================================
  AGENTIC RAG  —  maps to the BOTTOM diagram (11 steps)
================================================================

  What makes it "agentic": the LLM makes DECISIONS and can LOOP.
  It rewrites the query, decides IF it needs to retrieve, picks a
  SOURCE, then GRADES its own answer and retries if it's not good.

  (1)  Query
  (2)  Rewrite Query        -> node: rewrite_query
  (3)  Updated Query        -> stored in state
  (4)  Need More Details?   -> decision: decide_to_retrieve  (YES / NO)
  (5)  Which Source?        -> decision: route_to_source     (Vector DB / Internet)
  (6)  Sources              -> node: retrieve  (Vector DB  OR  Web search)
  (7)  Retrieved Context + Updated Query -> stored in state
  (8)  LLM                  -> node: generate
  (9)  Response             -> stored in state
  (10) Is the answer relevant? -> decision: grade_answer     (YES / NO -> loop to 2)
  (11) Final Response

  Free stack (same as basic RAG) + LangGraph for the control flow +
  DuckDuckGo as the free "Internet" source.

  Run:
    pip install -r requirements.txt
    #  GROQ_API_KEY=... in a .env file
    python 02_agentic_rag.py
"""

import os
from typing import List, TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ----------------------------------------------------------------------
#  Shared building blocks (same idea as basic RAG)
# ----------------------------------------------------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def build_retriever():
    docs = TextLoader("knowledge_base.md", encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80
    ).split_documents(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs.as_retriever(search_kwargs={"k": 3})


retriever = build_retriever()


def web_search(query: str) -> str:
    """The 'Internet' source (step 6). Free, no API key. Falls back gracefully."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=3))
        return "\n\n".join(h["body"] for h in hits) or "No web results found."
    except Exception as e:
        return f"(web search unavailable: {e})"


# ----------------------------------------------------------------------
#  The AGENT STATE  — the shared memory passed between every node.
#  Boxes 1,3,7,9 in the diagram all live here.
# ----------------------------------------------------------------------
class AgentState(TypedDict):
    question: str        # (1) original Query
    updated_query: str   # (3) rewritten query
    source: str          # (5) "vectorstore" or "web"
    context: str         # (7) retrieved context
    generation: str      # (9) LLM response
    retries: int         # loop guard for step (10)


# ======================================================================
#  NODES  (each = one box in the diagram)
# ======================================================================

def rewrite_query(state: AgentState) -> AgentState:
    """(2) Rewrite Query -> (3) Updated Query.
    Turn a raw/short question into a clearer, self-contained search query."""
    prompt = ChatPromptTemplate.from_template(
        "Rewrite the user's question into a clear, standalone search query. "
        "Return ONLY the rewritten query, nothing else.\n\nQuestion: {q}"
    )
    better = (prompt | llm | StrOutputParser()).invoke({"q": state["question"]})
    print(f"  (2) rewritten query -> {better!r}")
    return {"updated_query": better.strip()}


def retrieve(state: AgentState) -> AgentState:
    """(6) Sources -> (7) Retrieved Context.
    Pull context from whichever source step 5 chose."""
    q = state["updated_query"]
    if state["source"] == "web":
        print("  (6) retrieving from -> Internet (DuckDuckGo)")
        context = web_search(q)
    else:
        print("  (6) retrieving from -> Vector DB (FAISS)")
        docs = retriever.invoke(q)
        context = "\n\n".join(d.page_content for d in docs)
    return {"context": context}


def generate(state: AgentState) -> AgentState:
    """(8) LLM -> (9) Response. Answer grounded in whatever context we have."""
    context = state.get("context", "") or "(no external context was retrieved)"
    prompt = ChatPromptTemplate.from_template(
        "Answer the QUESTION using the CONTEXT when it is helpful.\n\n"
        "CONTEXT:\n{context}\n\nQUESTION: {q}\n\nANSWER:"
    )
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "q": state["question"]}
    )
    print("  (8) LLM generated an answer")
    return {"generation": answer}


# ======================================================================
#  DECISIONS  (the diamonds / dashed arrows in the diagram)
# ======================================================================

def decide_to_retrieve(state: AgentState) -> str:
    """(4) Need More Details?  YES -> pick a source ; NO -> answer directly."""
    prompt = ChatPromptTemplate.from_template(
        "Does answering this question REQUIRE looking up external facts, "
        "definitions, or data? Reply with only YES or NO.\n\nQuestion: {q}"
    )
    decision = (prompt | llm | StrOutputParser()).invoke(
        {"q": state["updated_query"]}
    ).strip().upper()
    need = decision.startswith("YES")
    print(f"  (4) Need more details? -> {'YES' if need else 'NO'}")
    return "retrieve" if need else "generate"


def route_to_source(state: AgentState) -> AgentState:
    """(5) Which Source? Choose the knowledge base vs the open internet."""
    prompt = ChatPromptTemplate.from_template(
        "You have two sources:\n"
        "- 'vectorstore': notes on ML, neural nets, transformers, RAG, embeddings, Groq.\n"
        "- 'web': anything else / current events / general facts.\n"
        "Which is best for this question? Reply with only one word: "
        "vectorstore or web.\n\nQuestion: {q}"
    )
    choice = (prompt | llm | StrOutputParser()).invoke(
        {"q": state["updated_query"]}
    ).strip().lower()
    source = "web" if "web" in choice else "vectorstore"
    print(f"  (5) Which source? -> {source}")
    return {"source": source}


def grade_answer(state: AgentState) -> str:
    """(10) Is the answer relevant?  YES -> finish (11) ; NO -> loop back to (2)."""
    # Safety valve so we never loop forever.
    if state["retries"] >= 2:
        print("  (10) max retries reached -> accepting answer")
        return "acceptable"

    prompt = ChatPromptTemplate.from_template(
        "Does the ANSWER actually and relevantly address the QUESTION? "
        "Reply with only YES or NO.\n\n"
        "QUESTION: {q}\nANSWER: {a}"
    )
    verdict = (prompt | llm | StrOutputParser()).invoke(
        {"q": state["question"], "a": state["generation"]}
    ).strip().upper()
    good = verdict.startswith("YES")
    print(f"  (10) Is the answer relevant? -> {'YES' if good else 'NO'}")
    return "acceptable" if good else "retry"


def bump_retries(state: AgentState) -> AgentState:
    """Tiny helper node: count a loop so grade_answer's safety valve works."""
    return {"retries": state["retries"] + 1}


# ======================================================================
#  BUILD THE GRAPH  (wire the boxes + arrows together)
# ======================================================================
def build_agent():
    g = StateGraph(AgentState)

    g.add_node("rewrite_query", rewrite_query)   # (2)
    g.add_node("route_to_source", route_to_source)  # (5)
    g.add_node("retrieve", retrieve)             # (6)+(7)
    g.add_node("generate", generate)             # (8)+(9)
    g.add_node("bump_retries", bump_retries)     # loop counter

    g.add_edge(START, "rewrite_query")           # (1) -> (2)

    # (4) Need More Details?  YES -> route to a source ; NO -> straight to LLM
    g.add_conditional_edges(
        "rewrite_query",
        decide_to_retrieve,
        {"retrieve": "route_to_source", "generate": "generate"},
    )

    g.add_edge("route_to_source", "retrieve")    # (5) -> (6)
    g.add_edge("retrieve", "generate")           # (7) -> (8)

    # (10) Is the answer relevant?  YES -> END (11) ; NO -> loop back to (2)
    g.add_conditional_edges(
        "generate",
        grade_answer,
        {"acceptable": END, "retry": "bump_retries"},
    )
    g.add_edge("bump_retries", "rewrite_query")  # loop: (10 NO) -> (2)

    return g.compile()


def main():
    agent = build_agent()
    print("\n=== AGENTIC RAG (Groq + LangChain + LangGraph) — 'exit' to quit ===\n")
    while True:
        question = input("You: ").strip()        # (1) Query
        if question.lower() in {"exit", "quit"}:
            break
        result = agent.invoke(
            {"question": question, "retries": 0}
        )
        print(f"\nBot (11) Final Response:\n{result['generation']}\n")


if __name__ == "__main__":
    main()
