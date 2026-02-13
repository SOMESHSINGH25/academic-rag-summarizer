import streamlit as st
from dotenv import load_dotenv
import os
import glob

from rag_pipeline import create_qa_chain, generate_questions, get_available_pdfs

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AcademIQ — Research Paper Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --cream:     #FAF8F3;
    --parchment: #F2EDE3;
    --warm-gray: #E8E0D0;
    --rule:      #C8BC9E;
    --ink:       #1A1612;
    --ink-mid:   #3D3529;
    --ink-light: #6B5F4E;
    --accent:    #8B1A1A;
    --accent-lt: #B5352E;
    --gold:      #C4922A;
    --gold-lt:   #E8B84B;
    --green:     #2A5C3F;
    --blue:      #1C3A5E;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--cream) !important;
    font-family: 'Source Serif 4', Georgia, serif;
    color: var(--ink);
}

[data-testid="stSidebar"] {
    background-color: var(--parchment) !important;
    border-right: 1px solid var(--rule);
}

[data-testid="stSidebar"] * {
    font-family: 'Source Serif 4', Georgia, serif !important;
}

/* ── Header ── */
.academiq-header {
    display: flex;
    align-items: baseline;
    gap: 14px;
    padding: 32px 0 8px 0;
    border-bottom: 2px solid var(--ink);
    margin-bottom: 6px;
}
.academiq-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.5px;
    line-height: 1;
}
.academiq-title span { color: var(--accent); }
.academiq-subtitle {
    font-family: 'Source Serif 4', serif;
    font-size: 0.95rem;
    color: var(--ink-light);
    font-style: italic;
    letter-spacing: 0.3px;
}
.academiq-rule {
    height: 1px;
    background: linear-gradient(to right, var(--gold), transparent);
    margin-bottom: 28px;
}

/* ── Section Labels ── */
.section-label {
    font-family: 'Playfair Display', serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--ink-light);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--warm-gray);
}

/* ── Answer Card ── */
.answer-card {
    background: white;
    border: 1px solid var(--warm-gray);
    border-left: 4px solid var(--accent);
    border-radius: 2px;
    padding: 24px 28px;
    margin: 16px 0;
    box-shadow: 0 2px 12px rgba(26,22,18,0.06);
    font-size: 1.02rem;
    line-height: 1.8;
    color: var(--ink-mid);
}

/* ── Source Badge ── */
.source-badge {
    display: inline-block;
    background: var(--parchment);
    border: 1px solid var(--rule);
    border-radius: 2px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--ink-light);
    margin: 4px 4px 4px 0;
}

/* ── Question Cards ── */
.q-card {
    background: white;
    border: 1px solid var(--warm-gray);
    border-radius: 2px;
    padding: 18px 22px;
    margin: 10px 0;
    box-shadow: 0 1px 6px rgba(26,22,18,0.05);
}
.q-number {
    font-family: 'Playfair Display', serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--ink-light);
    margin-bottom: 6px;
}
.q-text {
    font-size: 0.98rem;
    line-height: 1.65;
    color: var(--ink-mid);
}
.mcq-option {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 6px 0;
    font-size: 0.92rem;
    color: var(--ink-light);
}
.mcq-letter {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: var(--accent);
    min-width: 18px;
}

/* ── Sidebar PDF Card ── */
.pdf-info-card {
    background: white;
    border: 1px solid var(--warm-gray);
    border-radius: 2px;
    padding: 14px 16px;
    margin: 12px 0;
}
.pdf-name {
    font-family: 'Playfair Display', serif;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 4px;
    word-break: break-word;
}
.pdf-meta {
    font-size: 0.75rem;
    color: var(--ink-light);
    font-style: italic;
}

/* ── Tab Styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid var(--warm-gray);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 10px 24px !important;
    color: var(--ink-light) !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    border: 1px solid var(--warm-gray) !important;
    border-radius: 2px !important;
    background: white !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 0.97rem !important;
    color: var(--ink) !important;
    padding: 10px 14px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(139,26,26,0.08) !important;
}

/* ── Buttons ── */
.stButton button {
    font-family: 'Playfair Display', serif !important;
    font-size: 0.82rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    border: 1px solid var(--accent) !important;
    background: var(--accent) !important;
    color: white !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}
.stButton button:hover {
    background: var(--accent-lt) !important;
    border-color: var(--accent-lt) !important;
}

/* ── Selectbox ── */
.stSelectbox select, [data-baseweb="select"] {
    font-family: 'Source Serif 4', serif !important;
    border-radius: 2px !important;
}

/* ── Radio ── */
.stRadio label {
    font-family: 'Source Serif 4', serif !important;
    font-size: 0.93rem !important;
}

/* ── Spinner ── */
.stSpinner { color: var(--accent) !important; }

/* ── Divider ── */
hr { border-color: var(--warm-gray) !important; }

/* ── Chat History ── */
.chat-entry {
    padding: 12px 0;
    border-bottom: 1px solid var(--warm-gray);
}
.chat-q {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 4px;
}
.chat-a {
    font-size: 0.85rem;
    color: var(--ink-light);
    line-height: 1.6;
}

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 48px 24px;
    color: var(--ink-light);
    font-style: italic;
    font-size: 0.95rem;
    border: 1px dashed var(--rule);
    border-radius: 2px;
    margin: 24px 0;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Check API Key ───────────────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()


# ── Session State ──────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_pdf" not in st.session_state:
    st.session_state.selected_pdf = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 12px 0; border-bottom: 1px solid #C8BC9E; margin-bottom: 16px;'>
        <div style='font-family: Playfair Display, serif; font-size: 1.1rem; font-weight: 700; color: #1A1612;'>
            🎓 AcademIQ
        </div>
        <div style='font-size: 0.75rem; color: #6B5F4E; font-style: italic; margin-top: 2px;'>
            Research Paper Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">📂 Select Paper</div>', unsafe_allow_html=True)

    pdfs = get_available_pdfs()

    if not pdfs:
        st.warning("No PDFs found in data/samples/")
        st.info("Add PDF files to the data/samples/ folder to get started.")
    else:
        pdf_names = [os.path.basename(p) for p in pdfs]
        selected_name = st.selectbox(
            "Choose a paper:",
            options=pdf_names,
            label_visibility="collapsed"
        )

        selected_path = pdfs[pdf_names.index(selected_name)]

        # Show PDF info card
        file_size = os.path.getsize(selected_path) / 1024
        st.markdown(f"""
        <div class="pdf-info-card">
            <div class="pdf-name">📄 {selected_name}</div>
            <div class="pdf-meta">{file_size:.1f} KB</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📥 Load This Paper", use_container_width=True):
            with st.spinner("Loading paper into memory..."):
                try:
                    st.session_state.qa_chain = create_qa_chain(selected_path)
                    st.session_state.selected_pdf = selected_name
                    st.session_state.chat_history = []
                    st.success(f"✅ Loaded successfully!")
                except Exception as e:
                    st.error(f"Failed to load: {e}")

    st.markdown("---")

    # Chat History in sidebar
    if st.session_state.chat_history:
        st.markdown('<div class="section-label">🕑 History</div>', unsafe_allow_html=True)
        for i, entry in enumerate(reversed(st.session_state.chat_history[-5:])):
            st.markdown(f"""
            <div class="chat-entry">
                <div class="chat-q">Q: {entry['question'][:60]}{'...' if len(entry['question']) > 60 else ''}</div>
                <div class="chat-a">{entry['answer'][:100]}{'...' if len(entry['answer']) > 100 else ''}</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ── Main Content ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="academiq-header">
    <div class="academiq-title">Acad<span>em</span>IQ</div>
    <div class="academiq-subtitle">Intelligent Research Paper Analysis</div>
</div>
<div class="academiq-rule"></div>
""", unsafe_allow_html=True)

# Status bar
if st.session_state.selected_pdf:
    st.markdown(f"""
    <div style='font-size: 0.82rem; color: #6B5F4E; margin-bottom: 20px; font-style: italic;'>
        Currently analysing: <strong style='color: #1A1612;'>{st.session_state.selected_pdf}</strong>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='font-size: 0.82rem; color: #8B1A1A; margin-bottom: 20px; font-style: italic;'>
        ← Select and load a paper from the sidebar to begin
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Ask Questions", "Generate Questions"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASK QUESTIONS
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label" style="margin-top:20px;">Your Question</div>', unsafe_allow_html=True)

    query = st.text_input(
        "Question",
        placeholder="e.g. What is the main contribution of this paper?",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_btn = st.button("Ask →", use_container_width=True)

    if ask_btn:
        if not st.session_state.qa_chain:
            st.warning("⚠️ Please load a paper first using the sidebar.")
        elif not query.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("Analysing paper..."):
                try:
                    result = st.session_state.qa_chain.invoke({"input": query})
                    answer = result["answer"]

                    # Get source documents
                    sources = result.get("context", [])
                    source_pages = list(set([
                        f"Page {doc.metadata.get('page', '?') + 1}"
                        for doc in sources
                        if hasattr(doc, 'metadata')
                    ]))

                    # Save to history
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": answer,
                        "sources": source_pages
                    })

                    # Display answer
                    st.markdown('<div class="section-label" style="margin-top:24px;">Answer</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

                    # Show sources
                    if source_pages:
                        st.markdown('<div class="section-label" style="margin-top:16px;">Sources</div>', unsafe_allow_html=True)
                        badges = "".join([f'<span class="source-badge">📄 {p}</span>' for p in source_pages])
                        st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # Show full chat history
    if st.session_state.chat_history:
        st.markdown('<div class="section-label" style="margin-top:36px;">Previous Questions</div>', unsafe_allow_html=True)
        for entry in reversed(st.session_state.chat_history[:-1] if ask_btn else st.session_state.chat_history):
            with st.expander(f"Q: {entry['question'][:80]}"):
                st.markdown(f'<div class="answer-card">{entry["answer"]}</div>', unsafe_allow_html=True)
                if entry.get("sources"):
                    badges = "".join([f'<span class="source-badge">📄 {p}</span>' for p in entry["sources"]])
                    st.markdown(f'<div style="margin-top:8px;">{badges}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — GENERATE QUESTIONS
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label" style="margin-top:20px;">Question Type</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        q_type = st.radio(
            "Type",
            ["MCQ", "Short Answer", "Long Answer"],
            label_visibility="collapsed"
        )
    with col2:
        q_count = st.selectbox(
            "Number of questions",
            [3, 5, 7, 10],
            index=1,
            label_visibility="collapsed"
        )
    with col3:
        topic_hint = st.text_input(
            "Topic focus (optional)",
            placeholder="e.g. methodology, results...",
            label_visibility="collapsed"
        )

    gen_btn = st.button("Generate Questions →", use_container_width=False)

    if gen_btn:
        if not st.session_state.qa_chain:
            st.warning("⚠️ Please load a paper first using the sidebar.")
        else:
            with st.spinner(f"Generating {q_count} {q_type} questions..."):
                try:
                    questions = generate_questions(
                        st.session_state.qa_chain,
                        q_type=q_type,
                        count=q_count,
                        topic=topic_hint
                    )

                    st.markdown(f'<div class="section-label" style="margin-top:24px;">{q_type} Questions</div>', unsafe_allow_html=True)

                    if q_type == "MCQ":
                        for i, q in enumerate(questions):
                            st.markdown(f"""
                            <div class="q-card">
                                <div class="q-number">Question {i+1}</div>
                                <div class="q-text"><strong>{q.get('question', '')}</strong></div>
                                <div style="margin-top: 10px;">
                                    {''.join([f'<div class="mcq-option"><span class="mcq-letter">{opt["letter"]}.</span><span>{opt["text"]}</span></div>' for opt in q.get('options', [])])}
                                </div>
                                <div style="margin-top:10px; font-size:0.82rem; color:#2A5C3F; font-style:italic;">
                                    ✓ Answer: {q.get('answer', '')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    elif q_type == "Short Answer":
                        for i, q in enumerate(questions):
                            st.markdown(f"""
                            <div class="q-card">
                                <div class="q-number">Question {i+1}</div>
                                <div class="q-text">{q.get('question', '')}</div>
                                <div style="margin-top:10px; padding:10px; background:#FAF8F3; border-left:3px solid #C4922A; font-size:0.88rem; color:#3D3529; font-style:italic;">
                                    {q.get('answer', '')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    else:  # Long Answer
                        for i, q in enumerate(questions):
                            with st.expander(f"Question {i+1}: {q.get('question', '')[:70]}..."):
                                st.markdown(f'<div class="answer-card">{q.get("answer", "")}</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Error generating questions: {e}")
    else:
        if not st.session_state.qa_chain:
            st.markdown("""
            <div class="empty-state">
                Load a paper from the sidebar, then generate questions to test your understanding.
            </div>
            """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:0.75rem; color:#6B5F4E; font-style:italic; padding: 8px 0;'>
    AcademIQ · Built with LangChain + FAISS + Groq · For academic use
</div>
""", unsafe_allow_html=True)