# app.py
import os, re, hashlib, shutil, pandas as pd
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
CSV_PATH   = os.environ.get("NEWS_CSV", "articles.csv")   # your CSV path
INDEX_DIR  = os.environ.get("INDEX_DIR", "faiss_news_index")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Ollama (Llama 3) local server
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")

DATE_COL = "published_date"
AID_COL  = "id_article"  # your CSV id column
ALL_CATEGORIES = ["Technology","World","Politics","Science","Health","Sports","Culture","Entertainment","Society"]

# =========================
# GLOBAL CSS (dark, modern)
# =========================
st.set_page_config(page_title="News Agent (Local Llama3)", layout="wide")
st.markdown("""
<style>
:root{
  --bg:#0f1116; --bg-soft:#151822; --text:#EDF2F7; --muted:#8b91a1; --brand:#35c2c1; --border:#262b36;
  --card-grad1:#151a27; --card-grad2:#121723; --btn:#1e232e; --btn-hover:#242a37;
}
html, body, [data-testid="stAppViewContainer"]{ background:var(--bg); color:var(--text); }
[data-testid="stSidebar"]{ background:var(--bg-soft)!important; }
h1,h2,h3{ color:var(--text) }
a, a:visited{ color:var(--brand); text-decoration:none; }

.card-wrap{
  border:1px solid var(--border);
  background:linear-gradient(180deg,var(--card-grad1),var(--card-grad2));
  border-radius:18px; padding:16px; height:100%;
  box-shadow:0 0 0 1px rgba(53,194,193,.06), 0 10px 24px rgba(0,0,0,.35);
  transition: transform .12s ease, box-shadow .12s ease;
}
.card-wrap:hover{
  transform: translateY(-1px);
  box-shadow:0 0 0 1px rgba(53,194,193,.1), 0 14px 28px rgba(0,0,0,.40);
}
.card-title{ font-weight:700;font-size:1.05rem;line-height:1.3;margin:0 0 6px;color:var(--text); }
.card-meta{ color:var(--muted);font-size:.9rem;margin:0 0 12px; }
.card-link{ display:inline-block;color:var(--brand);font-weight:600;margin-bottom:12px; }
.card-link:hover{ text-decoration:underline; }

.card-actions { display:flex; gap:8px; margin-top:12px; }
button[kind="secondary"]{
  background:var(--btn); border:1px solid var(--border); color:var(--text);
  font-size:.85rem; padding:4px 8px; border-radius:10px; height:32px!important; width:100%;
}
button[kind="secondary"]:hover{
  background:var(--btn-hover); border-color:var(--brand); color:var(--brand); transform:translateY(-1px);
}

.section{ margin: 8px 0 22px; font-weight:800; font-size:1.25rem; letter-spacing:.2px; }
.section .tag{ color:var(--muted); font-weight:600; font-size:1rem; margin-left:.25rem; }
hr{ border-color:var(--border)!important }
.result-box{
  border:1px solid var(--border); background:linear-gradient(180deg,#151a27,#121723);
  border-radius:14px; padding:14px; margin:10px 0;
}
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

def item_keybase(item: Dict[str, Any]) -> str:
    if item.get("id"):  # prefer stable id
        return str(item["id"])[:10]
    raw = (item.get("url") or item.get("title") or "") + (item.get("published_date") or "")
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]

def render_card_with_actions(item: Dict[str, Any], keybase: str):
    title = item.get("title") or "(Untitled)"
    url   = item.get("url") or ""
    src   = item.get("source") or item.get("source_norm") or "Unknown source"
    dt    = item.get(DATE_COL)
    aid   = item.get("id","")

    # show whether content is present (based on lookup)
    have_text = False
    length = 0
    if aid and aid in ARTICLE_BY_ID:
        txt = str(ARTICLE_BY_ID[aid].get("content","") or "")
        have_text, length = (len(txt) > 0), len(txt)
    badge = "✅ full text" if have_text else "⚠️ no text"

    with st.container(border=False):
        st.markdown(f"""
        <div class="card-wrap">
          <div class="card-title">{title}</div>
          <div class="card-meta">{src} • {pretty_date(dt)}
            <span style="opacity:.6"> • id:{aid[:8]}</span> • {badge} (len={length})
          </div>
          <a class="card-link" href="{url}" target="_blank">Open article ↗</a>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card-actions">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Summarize", key=f"sum-{keybase}", use_container_width=True):
                st.session_state["pending_action"] = {"type": "summarize", "item": item}
        with col2:
            if st.button("🔎 Fact-check", key=f"fact-{keybase}", use_container_width=True):
                st.session_state["pending_action"] = {"type": "factcheck", "item": item}
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def doc_to_item(d: Document) -> Dict[str, Any]:
    m = d.metadata or {}
    return {
        "id": m.get("id", ""),  # carry id
        "title": m.get("title", "") or "(Untitled)",
        "url":   m.get("url", "") or "",
        "source": m.get("source", "") or m.get("source_norm", "") or "Unknown source",
        "published_date": m.get("published_date", ""),
    }

# ===== Global lookups (filled in load_retriever) =====
ARTICLE_BY_ID: Dict[str, Dict[str, Any]] = {}
ARTICLE_BY_URL: Dict[str, Dict[str, Any]] = {}

# =========================
# SUMMARIZER (FLAN-T5)
# =========================
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
import torch

@st.cache_resource(show_spinner=False)
def load_flan():
    model_name = "google/flan-t5-base"
    device = 0 if torch.cuda.is_available() else -1
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return hf_pipeline("text2text-generation", model=mdl, tokenizer=tok, device=device)

flan = load_flan()

def _sum_chunk(text: str, max_new_tokens: int = 180) -> str:
    prompt = (
        "Summarize the following news article in 3–5 concise bullet points, factual and neutral:\n\n"
        f"{text}"
    )
    out = flan(prompt, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)
    return out[0]["generated_text"].strip()

def summarize_text_long(text: str) -> str:
    if not text: return ""
    CHUNK = 1800  # chars
    chunks = [text[i:i+CHUNK] for i in range(0, len(text), CHUNK)]
    parts = [_sum_chunk(c, 160) for c in chunks]
    if len(parts) == 1:
        return parts[0]
    combined = "\n\n".join(parts)
    final = flan(
        "Combine these bullet-point summaries into 5–7 bullets, removing redundancy and keeping key facts:\n\n"
        + combined,
        max_new_tokens=200,
        temperature=0.0,
        do_sample=False
    )[0]["generated_text"].strip()
    return final

def get_article_text(item: Dict[str, Any]) -> str:
    aid = (item.get("id") or "").strip()
    if aid and aid in ARTICLE_BY_ID:
        return ARTICLE_BY_ID[aid].get("content","") or ""
    url = (item.get("url") or "").strip()
    if url and url in ARTICLE_BY_URL:
        return ARTICLE_BY_URL[url].get("content","") or ""
    return ""

def summarize_article(item: Dict[str, Any]) -> str:
    content = get_article_text(item)
    if not content:
        return "_No full text available in dataset to summarize._"
    try:
        summary = summarize_text_long(content)
        return f"**{item.get('title','(Untitled)')}**\n\n{summary}"
    except Exception as e:
        return f"_Summary failed: {e}_"

def factcheck_scaffold(item: Dict[str, Any]) -> str:
    content = get_article_text(item)
    if not content:
        return "_No full text available; cannot extract claims for checking._"
    prompt = (
        "From the article below, extract up to 3 checkable claims as short bullet points. "
        "For each claim, add one line 'How to verify:' with suggested authoritative sources "
        "(official stats, press releases, reputable outlets). Do NOT invent verdicts.\n\n"
        f"TITLE: {item.get('title','(Untitled)')}\n\nARTICLE:\n{content[:6000]}"
    )
    return ollama_chat(prompt, max_tokens=260, temperature=0.2)

# =========================
# RETRIEVER (FAISS)
# =========================
@st.cache_resource
def load_retriever():
    # build df
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV not found at {CSV_PATH}. Set NEWS_CSV or put articles.csv beside app.py.")
        st.stop()
    df = pd.read_csv(CSV_PATH).fillna("")

    # ensure base columns
    for col in ["title","content","url","source",DATE_COL,"category",AID_COL]:
        if col not in df.columns:
            df[col] = ""

    # id normalize (fallback if blank)
    def _ensure_id(val, row):
        s = str(val).strip()
        if s and s.lower() not in ("nan","none","null"):
            return s
        base = f"{row.get('url','')}|{row.get(DATE_COL,'')}|{row.get('title','')}"
        return hashlib.md5(base.encode("utf-8")).hexdigest()
    df[AID_COL] = df.apply(lambda r: _ensure_id(r.get(AID_COL, ""), r), axis=1)

    # source normalize
    def _canon(row):
        s = (row.get("source") or "").strip()
        if s: return s
        url = (row.get("url") or "").strip()
        try:
            host = urlparse(url).netloc.lower()
            return host[4:] if host.startswith("www.") else (host or "unknown")
        except: return "unknown"
    df["source_norm"] = df.apply(_canon, axis=1)

    # fast lookups
    article_by_id = {
        str(row[AID_COL]): {
            "id": str(row[AID_COL]),
            "title": str(row["title"]).strip(),
            "content": str(row["content"]).strip(),
            "url": str(row["url"]).strip(),
            "source": str(row["source"]).strip(),
            "source_norm": str(row["source_norm"]).strip(),
            "published_date": str(row[DATE_COL]).strip(),
            "category": str(row.get("category","")).strip(),
        }
        for _, row in df.iterrows()
    }
    article_by_url = {
        str(row["url"]).strip(): article_by_id[str(row[AID_COL])]
        for _, row in df.iterrows() if str(row.get("url","")).strip()
    }

    # FAISS docs (include id)
    docs = []
    for _, row in df.iterrows():
        meta = {k: str(row[k]) for k in df.columns if k != "content"}
        meta["id"] = str(row[AID_COL])
        docs.append(Document(page_content=f"{row['title']}\n\n{row['content']}", metadata=meta))

    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    os.makedirs(INDEX_DIR, exist_ok=True)
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_documents(docs, emb)
        vs.save_local(INDEX_DIR)

    # return everything you need on each run
    return vs.as_retriever(search_kwargs={"k": 12}), article_by_id, article_by_url

retriever, ARTICLE_BY_ID, ARTICLE_BY_URL = load_retriever()

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
        if domain in seen:
            continue
        seen.add(domain)
        out.append({
            "id": meta.get("id", ""),  # carry id
            "title": meta.get("title", ""),
            "source": meta.get("source", "") or meta.get("source_norm", ""),
            "published_date": meta.get("published_date", ""),
            "url": url
        })
        if len(out) >= k: break
    return out

# compatibility shim for retriever API
def _retrieve(query: str):
    try:
        return retriever.get_relevant_documents(query)
    except AttributeError:
        return retriever.invoke(query)

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

if st.sidebar.button("Get Top Articles"):
    by_cat = {}
    for cat in selected_categories:
        out = graph.invoke({"query": cat, "k": k, "mode": "", "results": [], "answer": ""})
        by_cat[cat] = out.get("results", [])
    st.session_state["articles_by_cat"] = by_cat

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
                render_card_with_actions(it, keybase=item_keybase(it))
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
    st.write(resp.get("answer", "No answer generated."))

    results = resp.get("results", []) or []
    if results:
        if isinstance(results[0], Document):
            top_items = [doc_to_item(d) for d in results[:3]]
        else:
            top_items = results[:3]

        st.markdown("<div class='section'>📰 Related Articles <span class='tag'>(top 3)</span></div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, it in enumerate(top_items):
            with cols[i % 3]:
                render_card_with_actions(it, keybase=item_keybase(it))

# ===== Handle card actions (summary / fact-check) =====
if "pending_action" in st.session_state:
    pa = st.session_state.pop("pending_action")
    aitem = pa["item"]

    if pa["type"] == "summarize":
        with st.spinner("Summarizing…"):
            summary = summarize_article(aitem)
        st.markdown("#### 📝 Summary")
        st.markdown(f"<div class='result-box'>{summary}</div>", unsafe_allow_html=True)

    elif pa["type"] == "factcheck":
        with st.spinner("Preparing fact-check checklist…"):
            notes = factcheck_scaffold(aitem)
        st.markdown("#### 🔎 Fact-check checklist (draft)")
        st.markdown(f"<div class='result-box'>{notes}</div>", unsafe_allow_html=True)
