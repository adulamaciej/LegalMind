import streamlit as st
from datasets import load_dataset
from pipeline.orchestrator import run_pipeline

st.set_page_config(
    page_title="LegalMind — ECHR Analysis",
    page_icon="⚖️",
    layout="wide"
)

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
            st.session_state['paragraphs'] = custom_text.split(". ")
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
            result = run_pipeline(st.session_state['paragraphs'])
        
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
        
        if verdict['violation']:
            st.error(f"⚠️ VIOLATION FOUND — Confidence: {verdict['confidence_score']}%")
        else:
            st.success(f"✅ NO VIOLATION — Confidence: {verdict['confidence_score']}%")
        
        st.write(f"**Violated Articles:** {', '.join(verdict.get('violated_articles', ['None']))}")
        st.write(f"**Reasoning:** {verdict['reasoning']}")
        
        if st.session_state.get('ground_truth'):
            st.divider()
            st.subheader("📊 Ground Truth (from dataset)")
            articles = ['2','3','5','6','8','9','10','11','14','P1-1']
            gt = [articles[l] for l in st.session_state['ground_truth']]
            st.write(f"**Actual violated articles:** {', '.join(gt)}")