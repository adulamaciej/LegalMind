import streamlit as st
from datasets import load_dataset
from config import ARTICLE_CODES
from pipeline.orchestrator import run_pipeline
import os

st.set_page_config(
    page_title="LegalMind — ECHR Analysis",
    page_icon="⚖️",
    layout="wide"
)

if not os.path.exists("./data/chroma"):
    st.info("First-time setup: indexing precedent database, this may take a few minutes...")
    from datasets import load_dataset
    from rag.indexer import get_collection, index_cases
    ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
    collection = get_collection()
    index_cases(list(ds['train']), collection)
    st.success("Setup complete!")

st.title("⚖️ LegalMind")
st.subheader("Multi-Agent ECHR Case Analysis System")

# Sidebar
st.sidebar.title("Options")
input_mode = st.sidebar.radio(
    "Input mode:",
    ["Use example from dataset", "Enter custom case"]
)

if input_mode == "Use example from dataset":
    case_id = st.sidebar.number_input(
        "Case ID (0-999):",
        min_value=0,
        max_value=999,
        value=0
    )
    
    if st.sidebar.button("Load Case"):
        ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
        case = ds['test'][case_id]
        st.session_state['paragraphs'] = case['text']
        st.session_state['ground_truth'] = case['labels']
        st.success(f"Case {case_id} loaded — {len(case['text'])} paragraphs")

else:
    custom_text = st.text_area(
        "Enter case facts:",
        height=300,
        placeholder="Enter the facts of the ECHR case here..."
    )
    
    if st.button("Submit Case"):
        if custom_text:
            st.session_state['paragraphs'] = [p.strip() for p in custom_text.split("\n") if p.strip()]
            st.session_state['ground_truth'] = None

# Główny panel
if 'paragraphs' in st.session_state:
    st.divider()
    
    # Pokaż fakty
    with st.expander("📄 Case Facts", expanded=False):
        for i, p in enumerate(st.session_state['paragraphs'][:5]):
            st.write(f"**{i+1}.** {p}")
        if len(st.session_state['paragraphs']) > 5:
            st.write(f"... and {len(st.session_state['paragraphs'])-5} more paragraphs")
    
    # Uruchom analizę
    if st.button("🔍 Analyze Case", type="primary"):
        with st.spinner("Running multi-agent analysis..."):
            try:
                result = run_pipeline(st.session_state['paragraphs'])
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()
        
        # Wyniki
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Extracted Facts")
            st.json(result['facts'])
        
        with col2:
            st.subheader("🔍 Precedents Found")
            for p in result['precedents']:
                st.write(f"**{p['id']}** — similarity: {p['similarity']:.3f}")
        
        st.divider()
        st.subheader("⚔️ Debate Transcript")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Prosecutor", "Defender", "Rebuttal", "Final Response"
        ])
        
        with tab1:
            st.write(result['debate']['prosecutor_arguments'])
        with tab2:
            st.write(result['debate']['defender_arguments'])
        with tab3:
            st.write(result['debate']['prosecutor_rebuttal'])
        with tab4:
            st.write(result['debate']['defender_final_response'])
        
        st.divider()
        st.subheader("👨‍⚖️ Verdict")
        
        verdict = result['verdict']
        
        if verdict.get('violation'):
            st.error(f"⚠️ VIOLATION FOUND — Confidence: {verdict.get('confidence_score', 'N/A')}%")
        else:
            st.success(f"✅ NO VIOLATION — Confidence: {verdict.get('confidence_score', 'N/A')}%")
        
        violated_articles = verdict.get('violated_articles') or ['None']
        st.write(f"**Violated Articles:** {', '.join(violated_articles)}")
        st.write(f"**Reasoning:** {verdict.get('reasoning', 'No reasoning provided')}")

        if st.session_state.get('ground_truth') is not None:
            st.divider()
            st.subheader("📊 Ground Truth (from dataset)")
            gt = [ARTICLE_CODES[l] for l in st.session_state['ground_truth']]
            st.write(f"**Actual violated articles:** {', '.join(gt)}")