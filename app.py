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

def get_genai_client():
    """Prioritizes Sidebar API Key, falls back to st.secrets."""
    if "api_key" in st.session_state and st.session_state["api_key"]:
        return genai.Client(api_key=st.session_state["api_key"])
    try:
        return genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
    except Exception:
        return None

@st.cache_resource
def init_resources():
    """Initializes embeddings and an ephemeral (in-memory) Chroma client."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client_db = chromadb.EphemeralClient()
    return client_db, ef

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

# --- 2. GRAPH NODES ---
def retrieval_node(state: InsuranceState):
    """Semantic search + Page-X metadata filtering."""
    query = state['query'].lower()
    try:
        collection = chroma_client.get_collection(name="insurance_vault", embedding_function=default_ef)
    except Exception:
        return {"relevant_chunks": "⚠️ NO DOCUMENTS INGESTED."}

    # "Go to Page X" Regex
    page_match = re.search(r'(?:page|pg|p\.?)\s*(\d+)', query)
    
    if page_match:
        target_page = int(page_match.group(1))
        results = collection.get(where={"page": target_page}, limit=5)
    else:
        results = collection.query(query_texts=[state['query']], n_results=12)

    context = ""
    docs = results.get('documents', [])
    metas = results.get('metadatas', [])

    if docs and isinstance(docs[0], list):
        docs, metas = docs[0], metas[0]

    for i in range(len(docs)):
        source = metas[i]['source']
        page = metas[i]['page']
        context += f"--- SOURCE: {source} (Page {page}) ---\n{docs[i]}\n\n"
        
    return {"relevant_chunks": context}

def analyzer_node(state: InsuranceState):
    """Clinical Senior Auditor logic including User Profile analysis."""
    client = get_genai_client()
    if not client:
        return {"final_response": "⚠️ Please provide an API Key in the sidebar or secrets.", "needs_search": False}

    prompt = f"""
    SYSTEM: You are a "Context-Only" Senior Insurance Auditor (Alternative to Ditto/Beshak). 
    Your goal is to provide clinical, high-precision audits of insurance policies.

    STRICT OPERATING RULES:
    1.  **Source Grounding:** Answer ONLY using 'PDF FRAGMENTS' or 'WEB SEARCH DATA'. Do not use internal knowledge.
    2.  **Missing Info:** If the data is not in the fragments or web results, you MUST say: "I'm sorry, the provided documents and web data do not contain this information."
    3.  **Citations:** Every claim must be cited. 
        - For PDFs: [Source: Filename, Page X]
        - For Web: [Source: Web Search Data]
    4.  **Multi-PDF Logic:** If multiple sources are present, clearly distinguish between them (e.g., "HDFC offers X, whereas Care Supreme offers Y").
    5.  **Internet Trigger:** If 'use_internet' is True and the PDF doesn't have the answer, set 'needs_search' to True and generate a specific 'search_query'.
    6.  **Scope Guardrail:** If the query is unrelated to insurance, medical policies, or health coverage, politely state: "This query falls outside the scope of a professional insurance audit."
    7.  **Calculation Integrity:** If the user asks for a 'Plus' benefit or 'Secure' multiplier, perform the math based strictly on the percentage defined in the document.
    8.  **Personalization:** Evaluate policy features against the 'USER PROFILE'. If a policy exceeds the budget or misses a requirement, highlight this.

    USER PROFILE: 
    - Budget: {state['budget']}
    - Requirements/Needs: {state['requirements']}

    PDF FRAGMENTS: 
    {state['relevant_chunks']}

    WEB SEARCH DATA: 
    {state['search_results']}

    QUERY: 
    {state['query']}
    """

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
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
    """DuckDuckGo Web Search Tool."""
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
workflow.add_conditional_edges(
    "analyzer",
    lambda x: "searcher" if x["needs_search"] and x["iterations"] < 2 else END
)
workflow.add_edge("searcher", "retriever")
app = workflow.compile()

# --- 4. STREAMLIT UI ---
st.title("🛡️ InsurAudit AI")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    key_input = st.text_input("API Key Override", type="password", help="Overrides st.secrets if provided.")
    if key_input: st.session_state["api_key"] = key_input
    
    st.divider()
    st.subheader("User Profile")
    user_budget = st.number_input("Annual Budget (INR)", min_value=0, value=15000)
    user_reqs = st.text_area("Specific Requirements", placeholder="e.g., Maternity cover, No room rent cap").split(",")
    
    st.divider()
    use_internet = st.toggle("Enable Web Search", value=True)
    multi_pdf = st.toggle("Multi-PDF Audit", value=True)
    
    st.divider()
    uploaded_files = st.file_uploader("Upload Policy Documents", accept_multiple_files=True, type=['pdf'])
    
    if st.button("📥 Ingest Documents"):
        if not uploaded_files:
            st.error("Please upload PDFs first.")
        else:
            with st.spinner("Processing Documents..."):
                try: chroma_client.delete_collection("insurance_vault")
                except: pass
                
                collection = chroma_client.create_collection(name="insurance_vault", embedding_function=default_ef)
                to_process = uploaded_files if multi_pdf else [uploaded_files[0]]
                
                for up_file in to_process:
                    with tempfile.NamedTemporaryFile(delete=False) as tf:
                        tf.write(up_file.getbuffer())
                        doc = fitz.open(tf.name)
                        for page in doc:
                            text = page.get_text().strip()
                            if len(text) > 50:
                                collection.add(
                                    documents=[text],
                                    metadatas=[{"source": up_file.name, "page": page.number + 1}],
                                    ids=[f"{up_file.name}_pg_{page.number + 1}"]
                                )
                st.success(f"Audit vault updated with {len(to_process)} files.")

# --- CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Verify policy rules..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        ui_status = st.status("Initializing Audit...")
        
        inputs = {
            "query": query, "relevant_chunks": "", "budget": user_budget, 
            "requirements": [r.strip() for r in user_reqs if r.strip()],
            "search_results": "", "citations": [], "needs_search": False,
            "use_internet": use_internet, "iterations": 0
        }

        final_out = ""
        for output in app.stream(inputs):
            for node, data in output.items():
                if node == "searcher": ui_status.update(label="🔍 Performing Web Audit...")
                if node == "retriever": ui_status.update(label="📖 Searching Policy Fragments...")
                if node == "analyzer" and not data.get("needs_search"):
                    final_out = data['final_response']
        
        ui_status.update(label="✅ Audit Complete", state="complete")
        st.markdown(final_out)
        st.session_state.messages.append({"role": "assistant", "content": final_out})
