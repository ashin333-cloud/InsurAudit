import streamlit as st
import os
import fitz  # PyMuPDF
import json
import chromadb
import re
import tempfile
import warnings
from chromadb.utils import embedding_functions
from typing import List, TypedDict
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END
from ddgs import DDGS

# --- INITIALIZATION ---
warnings.filterwarnings("ignore", message="datetime.datetime.utcnow")

st.set_page_config(page_title="InsurAudit AI", page_icon="🛡️", layout="wide")

@st.cache_resource
def init_resources():
    # Using the exact embedding model from your snippet
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    # Persistent client to handle Streamlit re-runs better
    client = chromadb.PersistentClient(path="./insur_audit_db")
    return client, ef

chroma_client, default_ef = init_resources()

# --- 1. STATE DEFINITION ---
class InsuranceState(TypedDict):
    query: str
    relevant_chunks: str
    budget: float
    requirements: List[str]
    search_results: str
    final_response: str
    citations: List[int]
    needs_search: bool
    use_internet: bool
    iterations: int

# --- 2. CORE FUNCTIONS (Your logic) ---
def retrieval_node(state: InsuranceState):
    query = state['query'].lower()
    collection = chroma_client.get_collection(name="insurance_vault", embedding_function=default_ef)
    
    # "Go to Page X" Logic
    page_match = re.search(r'(?:page|pg|p\.?)\s*(\d+)', query)
    if page_match:
        target_page = int(page_match.group(1))
        results = collection.get(where={"page": target_page}, limit=5)
    else:
        results = collection.query(query_texts=[state['query']], n_results=12)

    context = ""
    docs = results['documents'] if 'documents' in results else []
    metas = results['metadatas'] if 'metadatas' in results else []

    if docs and isinstance(docs[0], list):
        docs, metas = docs[0], metas[0]

    for i in range(len(docs)):
        source = metas[i]['source']
        page = metas[i]['page']
        context += f"--- SOURCE: {source} (Page {page}) ---\n{docs[i]}\n\n"
    return {"relevant_chunks": context}

def analyzer_node(state: InsuranceState):
    prompt = f"""
    SYSTEM: You are a "Context-Only" Senior Insurance Auditor (Alternative to Ditto/Beshak).
    STRICT RULES:
    1. Answer ONLY using 'PDF FRAGMENTS' or 'WEB SEARCH DATA'. 
    2. If info is missing, you MUST say: "I'm sorry, the provided documents do not contain this information."
    3. Cite page numbers for every PDF-based claim (e.g., [Source: Filename, Page 5]).
    4. If 'use_internet' is True and info is missing, set 'needs_search' to True.
    5. If the query is completely unrelated to insurance/medical field, refuse politely.

    PDF FRAGMENTS: {state['relevant_chunks']}
    WEB SEARCH DATA: {state['search_results']}
    QUERY: {state['query']}
    """
    
    api_key = st.session_state.get("api_key", "")
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview", # KEPT AS IS
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "answer": {"type": "STRING"},
                    "citations": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "needs_search": {"type": "BOOLEAN"},
                    "search_query": {"type": "STRING"}
                },
                "required": ["answer", "citations", "needs_search"]
            }
        )
    )
    res = json.loads(response.text)
    return {
        "final_response": res['answer'],
        "citations": res['citations'],
        "needs_search": res['needs_search'] if state['use_internet'] else False,
        "query": res.get('search_query', state['query']),
        "iterations": state['iterations'] + 1
    }

def search_node(state: InsuranceState):
    with DDGS() as ddgs:
        results = [r['body'] for r in ddgs.text(state['query'], max_results=5)]
        return {"search_results": "\n".join(results), "needs_search": False}

# --- 3. GRAPH CONSTRUCTION ---
workflow = StateGraph(InsuranceState)
workflow.add_node("retriever", retrieval_node)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("searcher", search_node)
workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "analyzer")
workflow.add_conditional_edges("analyzer", lambda x: "searcher" if x["needs_search"] and x["iterations"] < 2 else END)
workflow.add_edge("searcher", "retriever")
app = workflow.compile()

# --- 4. STREAMLIT UI ---
st.title("🛡️ InsurAudit AI")

with st.sidebar:
    st.header("⚙️ Configuration")
    key = st.text_input("Enter Google API Key", type="password")
    if key: st.session_state["api_key"] = key
    
    use_internet = st.toggle("Enable Internet Search", value=True)
    multi_pdf = st.toggle("Enable Multi-PDF Focus", value=True)
    
    st.divider()
    uploaded_files = st.file_uploader("Upload Policy PDFs", accept_multiple_files=True, type=['pdf'])
    
    if st.button("📥 Ingest Documents"):
        if not uploaded_files:
            st.error("Please upload a PDF first.")
        elif not key:
            st.error("Please enter an API key.")
        else:
            with st.spinner("Processing documents..."):
                try: chroma_client.delete_collection("insurance_vault")
                except: pass
                col = chroma_client.create_collection(name="insurance_vault", embedding_function=default_ef)
                
                # Logic: focus only on first if multi_pdf is False
                to_process = uploaded_files if multi_pdf else [uploaded_files[0]]
                
                for up_file in to_process:
                    with tempfile.NamedTemporaryFile(delete=False) as tf:
                        tf.write(up_file.getbuffer())
                        doc = fitz.open(tf.name)
                        for page in doc:
                            text = page.get_text().strip()
                            if len(text) > 50:
                                col.add(
                                    documents=[text],
                                    metadatas=[{"source": up_file.name, "page": page.number + 1}],
                                    ids=[f"{up_file.name}_pg_{page.number + 1}"]
                                )
                st.success(f"Successfully ingested {len(to_process)} PDF(s).")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about the policy rules..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        status = st.status("Analyzing...")
        
        inputs = {
            "query": query, "relevant_chunks": "", "budget": 0, "requirements": [], 
            "search_results": "", "citations": [], "needs_search": False, 
            "use_internet": use_internet, "iterations": 0
        }

        final_out = ""
        for output in app.stream(inputs):
            for node, data in output.items():
                if node == "searcher": status.update(label="🔍 Searching web...")
                if node == "retriever": status.update(label="📖 Reading PDF...")
                if node == "analyzer" and not data.get("needs_search"):
                    final_out = data['final_response']
        
        status.update(label="✅ Audit Complete", state="complete")
        st.markdown(final_out)
        st.session_state.messages.append({"role": "assistant", "content": final_out})
