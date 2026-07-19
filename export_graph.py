"""
Export the Agentic RAG LangGraph as a picture you can show next to the slide.

Produces:
  - agentic_rag_graph.mmd  (Mermaid text — always works, paste into mermaid.live)
  - agentic_rag_graph.png  (rendered image — needs internet OR `pip install pygraphviz`)
  - an ASCII sketch printed to the terminal (always works, great for a quick screen-share)

Run:
    python export_graph.py
"""

from importlib import import_module

# Reuse the exact graph you built in 02_agentic_rag.py — single source of truth.
agent = import_module("02_agentic_rag").build_agent()
graph = agent.get_graph()

# 1) Mermaid text — zero dependencies, never fails.
mermaid = graph.draw_mermaid()
with open("agentic_rag_graph.mmd", "w", encoding="utf-8") as f:
    f.write(mermaid)
print("[ok] wrote agentic_rag_graph.mmd  (paste it into https://mermaid.live)")

# 2) ASCII sketch — nice for a terminal screen-share.
print("\n----- graph (ASCII) -----")
try:
    print(graph.draw_ascii())
except Exception as e:
    print("(ascii needs `pip install grandalf`):", e)

# 3) PNG — best-looking, needs internet (mermaid.ink) or a local graphviz.
try:
    png = graph.draw_mermaid_png()
    with open("agentic_rag_graph.png", "wb") as f:
        f.write(png)
    print("\n[ok] wrote agentic_rag_graph.png")
except Exception as e:
    print("\n[skip] PNG render unavailable (need internet or pygraphviz):", e)
    print("       Use the .mmd file at https://mermaid.live instead.")
