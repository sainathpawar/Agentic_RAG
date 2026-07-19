"""
============================================================
  BASIC RAG  —  maps 1:1 to the TOP diagram (7 steps)
============================================================

  (1) Query
  (2) Encode      -> turn text into vectors  (Embeddings model)
  (3) Index       -> store vectors in a Vector Database (FAISS)
  (4) Similarity Search -> find the most similar docs
  (5) Retrieved Context + Query -> stuff docs into a prompt
  (6) LLM         -> Groq (Llama 3) reads context + query
  (7) Response    -> grounded answer

  Free stack:
    - LLM        : Groq API (free tier)          -> ChatGroq
    - Embeddings : HuggingFace MiniLM (LOCAL)    -> no API key, runs on your PC
    - Vector DB  : FAISS (LOCAL)                 -> no server needed

  Run:
    pip install -r requirements.txt
    #  put GROQ_API_KEY=... in a .env file (get key at https://console.groq.com )
    python 01_basic_rag.py
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()  # reads GROQ_API_KEY from a .env file


# ----------------------------------------------------------------------
# STEP 2 (part a): the EMBEDDINGS model
# ----------------------------------------------------------------------
# NOTE the diagram labels this box "LLM", but in practice
# the ENCODER is a dedicated *embeddings* model, not the chat LLM.
# We use a small, free, LOCAL model — nothing is sent to any paid API.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def build_vector_store():
    """STEPS 2 + 3: Encode all documents and INDEX them in FAISS."""
    # Load our knowledge base file...
    docs = TextLoader("knowledge_base.md", encoding="utf-8").load()

    # ...and split it into small overlapping chunks so retrieval is precise.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    print(f"[index] split knowledge base into {len(chunks)} chunks")

    # FAISS.from_documents() does BOTH:
    #   (2) encode every chunk into a vector
    #   (3) index those vectors for fast search
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def build_rag_chain(vector_store):
    """Wire steps 4 -> 5 -> 6 -> 7 into one LangChain pipeline."""

    # STEP 4: SIMILARITY SEARCH — the retriever fetches the top-k similar chunks.
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # STEP 6: the LLM — Groq running Llama 3 (free & fast).
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # STEP 5: the PROMPT — combine "Retrieved Context" + the user's "Query".
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful teaching assistant.
Answer the QUESTION using ONLY the CONTEXT below.
If the answer is not in the context, say "I don't know from the given context."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # The full chain: retrieve -> build prompt -> call LLM -> parse text.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt          # STEP 5: context + query
        | llm             # STEP 6: LLM
        | StrOutputParser()  # STEP 7: clean text response
    )
    return rag_chain


def main():
    vector_store = build_vector_store()
    rag_chain = build_rag_chain(vector_store)

    print("\n=== BASIC RAG (Groq + LangChain) — type 'exit' to quit ===\n")
    while True:
        query = input("You: ").strip()          # STEP 1: Query
        if query.lower() in {"exit", "quit"}:
            break
        answer = rag_chain.invoke(query)          # STEP 7: Response
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()
