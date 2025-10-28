# app.py
import os, re, pandas as pd
import streamlit as st
from urllib.parse import urlparse
from typing import TypedDict, List, Dict, Any
from datetime import datetime
import requests

from langgraph.graph import StateGraph, END
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# =========================
# CONFIG
# =========================
CSV_PATH   = os.environ.get("NEWS_CSV", "articles.csv")   # put your CSV here
INDEX_DIR  = os.environ.get("INDEX_DIR", "faiss_news_index")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Ollama (Llama 3) local server
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")

DATE_COL = "published_date"
ALL_CATEGORIES = ["Technology","World","Politics","Science","Health","Sports","Culture","Entertainment","Society"]

# =========================
# GLOBAL CSS (dark cards)
# =========================
st.set_page_config(page_title="News Agent (Local Llama3)", layout="wide")
st.markdown("""
<style>
:root{
  --bg:#0f1116; --bg-soft:#151822; --text:#e6e9ef; --muted:#8b91a1; --brand:#35c2c1; --border:#2a2f3b;
}
html, body, [data-testid="stAppViewContainer"]{ background:var(--bg); color:var(--text); }
[data-testid="stSidebar"]{ background:var(--bg-soft)!important; }
h1,h2,h3{ color:var(--text) }
a, a:visited{ color:var(--brand); text-decoration:none; }

.card{
  border:1px solid var(--border);
  background:linear-gradient(180deg,#141826,#121623);
  border-radius:18px; padding:16px; height:100%;
  box-shadow:0 0 0 1px rgba(53,194,193,.08), 0 8px 24px rgba(0,0,0,.35);
}
.card .title{
  font-weight:700;font-size:1.05rem;line-height:1.3;margin:0 0 6px;color:var(--text);
}
.card .meta{
  color:var(--muted);font-size:.9rem;margin:0 0 10px;
}
.card a.open{
  display:inline-block;color:var(--brand);font-weight:600;
}
.card a.open:hover{ text-decoration:underline; }

.section{ margin: 8px 0 22px; font-weight:800; font-size:1.25rem; letter-spacing:.2px; }
.section .tag{ color:var(--muted); font-weight:600; font-size:1rem; margin-left:.25rem; }
hr{ border-color:var(--border)!important }
</style>
""", unsafe_allow_html=True)

st.title("📰 News AI Agent — Local Llama 3 (Ollama)")

# =========================
# HELPERS
# =========================
def pretty_date(date_str):
    if not date_str or str(date_str) in ("nan","NaT"): return "Unknown date"
    try:
        dt = datetime.fromisoformat(str(date_str).replace("Z","+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(date_str)

def article_card(item: dict):
    title = item.get("title") or "(Untitled)"
    url   = item.get("url") or ""
    src   = item.get("source") or item.get("source_norm") or "Unknown source"
    dt    = item.get(DATE_COL)
    st.markdown(f"""
      <div class="card">
        <div class="title">{title}</div>
        <div class="meta">{src} • {pretty_date(dt)}</div>
        <a class="open" href="{url}" target="_blank">Open article ↗</a>
      </div>
    """, unsafe_allow_html=True)

def ollama_chat(prompt: str, max_tokens: int = 220, temperature: float = 0.3) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "").strip()

# =========================
# RETRIEVER (FAISS)
# =========================
@st.cache_resource
def load_retriever():
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV not found at {CSV_PATH}. Set NEWS_CSV or put articles.csv beside app.py.")
        st.stop()
    df = pd.read_csv(CSV_PATH).fillna("")
    # ensure base columns exist
    for col in ["title","content","url","source",DATE_COL,"category"]:
        if col not in df.columns: df[col] = ""
    # normalize source for diversity
    def _canon(row):
        s = (row.get("source") or "").strip()
        if s: return s
        url = (row.get("url") or "").strip()
        try:
            host = urlparse(url).netloc.lower()
            return host[4:] if host.startswith("www.") else (host or "unknown")
        except: return "unknown"
    df["source_norm"] = df.apply(_canon, axis=1)

    docs = [
        Document(
            page_content=f"{row['title']}\n\n{row['content']}",
            metadata={k: str(row[k]) for k in df.columns if k != "content"},
        )
        for _, row in df.iterrows()
    ]
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    os.makedirs(INDEX_DIR, exist_ok=True)
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_documents(docs, emb)
        vs.save_local(INDEX_DIR)
    return vs.as_retriever(search_kwargs={"k": 12})

retriever = load_retriever()

# =========================
# AGENT (LangGraph)
# =========================
class State(TypedDict):
    query: str
    k: int
    mode: str
    results: List[Dict[str, Any]] | List[Document]
    answer: str

def format_hits(hits: List[Document], k: int) -> List[Dict[str, Any]]:
    out, seen = [], set()
    for d in hits:
        meta = d.metadata or {}
        url = meta.get("url", "")
        domain = urlparse(url).netloc
        if domain in seen:  # 1/article per domain (diverse sources)
            continue
        seen.add(domain)
        out.append({
            "title": meta.get("title", ""),
            "source": meta.get("source", "") or meta.get("source_norm", ""),
            "published_date": meta.get("published_date", ""),
            "url": url
        })
        if len(out) >= k: break
    return out

# --- compatibility shim for retriever API ---
def _retrieve(query: str):
    """Return a list of Documents; supports both old and new LangChain retriever APIs."""
    try:
        return retriever.get_relevant_documents(query)   # older API
    except AttributeError:
        return retriever.invoke(query)                   # newer Runnable retrievers

def route(state: State) -> str:
    q = (state["query"] or "").lower().strip()
    if q.endswith("?") or re.match(r"^(what|who|why|how|when|where)\b", q):
        state["mode"] = "qa"; return "qa"
    state["mode"] = "list"; return "list"

def router_node(state: State) -> State:
    route(state); return state

def node_retrieve_list(state: State) -> State:
    docs = _retrieve(state["query"])
    state["results"] = format_hits(docs, state["k"])
    return state

def node_retrieve_qa(state: State) -> State:
    docs = _retrieve(state["query"])
    state["results"] = docs[:6]
    return state

def node_llm_answer(state: State) -> State:
    docs: List[Document] = state["results"]
    if not docs:
        state["answer"] = "I couldn't find relevant articles to answer that."
        return state
    # keep context compact for faster, cheaper generations
    max_docs, max_chars = 4, 350
    context = []
    for d in docs[:max_docs]:
        m = d.metadata or {}
        title = m.get("title","")
        snippet = d.page_content[:max_chars]
        context.append(f"[{title}] {snippet}")
    ctx = "\n\n".join(context)

    prompt = (
        "You are a concise journalist assistant.\n"
        "Use ONLY the CONTEXT to answer the user's question. "
        "Cite article titles in parentheses when relevant.\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"QUESTION: {state['query']}\n\n"
        "ANSWER:"
    )
    state["answer"] = ollama_chat(prompt, max_tokens=220, temperature=0.3)
    return state

# Build graph
g = StateGraph(State)
g.add_node("router", router_node)
g.set_entry_point("router")
g.add_conditional_edges("router", route, {"list": "retrieve_list", "qa": "retrieve_qa"})
g.add_node("retrieve_list", node_retrieve_list)
g.add_node("retrieve_qa", node_retrieve_qa)
g.add_node("llm_answer", node_llm_answer)
g.add_edge("retrieve_list", END)
g.add_edge("retrieve_qa", "llm_answer")
g.add_edge("llm_answer", END)
graph = g.compile()

# =========================
# UI
# =========================
with st.sidebar:
    st.header("Filters")
    selected_categories = st.multiselect("Categories", ALL_CATEGORIES, default=["Technology","Politics"])
    k = st.slider("Articles per category", 1, 10, 5)
    show_table = st.checkbox("Show raw table", value=False)

# Fetch per-category
if st.sidebar.button("Get Top Articles"):
    by_cat = {}
    for cat in selected_categories:
        out = graph.invoke({"query": cat, "k": k, "mode": "", "results": [], "answer": ""})
        by_cat[cat] = out.get("results", [])
    st.session_state["articles_by_cat"] = by_cat

# Render results
if "articles_by_cat" in st.session_state:
    for cat, items in st.session_state["articles_by_cat"].items():
        st.markdown(
            f"<div class='section'>{cat} — {len(items)} most recent <span class='tag'>(diverse sources)</span></div>",
            unsafe_allow_html=True,
        )
        if not items:
            st.caption("No articles found.")
            continue
        cols = st.columns(3)
        for i, it in enumerate(items):
            with cols[i % 3]:
                article_card(it)
        st.markdown("<hr/>", unsafe_allow_html=True)
        if show_table:
            st.dataframe(pd.DataFrame(items))

st.divider()
st.subheader("💬 Ask a Question")
user_q = st.text_input("Your question about the news:")
if user_q:
    with st.spinner("Thinking..."):
        resp = graph.invoke({"query": user_q, "k": k, "mode": "", "results": [], "answer": ""})
        st.markdown("### 🧠 Answer")
        st.write(resp["answer"])
