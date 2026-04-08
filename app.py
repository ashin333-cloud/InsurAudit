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
    query = state['query'].lower()
    try:
        collection = chroma_client.get_collection(name="insurance_vault", embedding_function=default_ef)
    except Exception:
        return {"relevant_chunks": "⚠️ NO DOCUMENTS INGESTED."}

    # INCREASE n_results to 25 to ensure we catch the Brochure too
    results = collection.query(
        query_texts=[state['query']], 
        n_results=25  # Expanded horizon
    )

    context = ""
    docs = results.get('documents', [[]])[0]
    metas = results.get('metadatas', [[]])[0]

    # TRACKING: Let's see which files we found
    found_sources = set()
    
    for i in range(len(docs)):
        source = metas[i]['source']
        found_sources.add(source)
        page = metas[i]['page']
        context += f"--- SOURCE: {source} (Page {page}) ---\n{docs[i]}\n\n"
    
    # DEBUG PRINT (Visible in your Streamlit terminal/logs)
    print(f"Sources retrieved for query: {found_sources}")
        
    return {"relevant_chunks": context}

def analyzer_node(state: InsuranceState):
    """Clinical Senior Auditor logic including User Profile analysis."""
    client = get_genai_client()
    if not client:
        return {"final_response": "⚠️ Please provide an API Key in the sidebar or secrets.", "needs_search": False}

    prompt = f"""
    SYSTEM: You are a "Context-Only" Senior Insurance Auditor. 
    Your goal is to provide clinical, high-precision audits in a SPLENDID, professional format.

    FORMATTING RULES:
    1. Use ## Headings for major sections (e.g., ## Policy Overview).
    2. Use **Bold text** for key terms and limits.
    3. Use Bullet points for lists of benefits or exclusions.
    4. Use a Table if comparing multiple policies or complex numbers.
    5. Ensure the response is scannable and visually organized.

    STRICT OPERATING RULES:
    1. Source Grounding: Answer ONLY using 'PDF FRAGMENTS' or 'WEB SEARCH DATA'.
    2. Missing Info: If data is missing, say: "I'm sorry, the provided documents do not contain this information."
    3. Citations: Every claim MUST be cited as [Source: Filename, Page X].
    4. HARD Scope Guardrail: - ALLOWED: Policy summaries, coverage explanations, medical limits, and **insurer performance metrics (e.g., Claim Settlement Ratios, Solvency Ratios, Network Hospital lists).**
       - FORBIDDEN: Biographies, movies, general trivia, or unrelated business advice.
       - REJECTION LOGIC: Use the "Outside professional audit scope" phrase ONLY for truly unrelated topics.
    5. Internet Trigger: If 'use_internet' is True and the PDF fragments do not contain the SPECIFIC FACT requested (e.g., a list of names, a specific % or a fee), you MUST set 'needs_search' to True and generate a 'search_query'. Do not simply refer the user to a website if you have the power to search it.
    
    USER PROFILE: 
    - Budget: {state['budget']}
    - Requirements: {state['requirements']}

    PDF FRAGMENTS: {state['relevant_chunks']}
    WEB SEARCH DATA: {state['search_results']}
    QUERY: {state['query']}
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
        ui_status = st.status("🔍 Initializing Audit...")
        
        inputs = {
            "query": query, "relevant_chunks": "", "budget": user_budget, 
            "requirements": [r.strip() for r in user_reqs if r.strip()],
            "search_results": "", "citations": [], "needs_search": False,
            "use_internet": use_internet, "iterations": 0
        }

        final_out = ""
        # We iterate through the Graph stream
        for output in app.stream(inputs):
            for node, data in output.items():
                if node == "retriever":
                    ui_status.update(label="📖 Searching Policy Fragments...")
                
                if node == "analyzer":
                    if data.get("needs_search"):
                        ui_status.update(label="⚠️ No result found in PDFs. Switching to Internet...")
                    else:
                        final_out = data.get('final_response', "")

                if node == "searcher":
                    ui_status.update(label="🌐 Searching live web data for: " + data.get("query", query))
        
        ui_status.update(label="✅ Audit Complete", state="complete")
        st.markdown(final_out)
        st.session_state.messages.append({"role": "assistant", "content": final_out})
