import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DATA_DIR = "data/samples"
VECTOR_DB_BASE = "vectorstore"


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_available_pdfs():
    """Return list of PDF paths in data/samples/"""
    return sorted(Path(DATA_DIR).glob("*.pdf"))


def get_vectorstore_path(pdf_path: str) -> str:
    """Each PDF gets its own vectorstore folder"""
    name = Path(pdf_path).stem
    return os.path.join(VECTOR_DB_BASE, name)


def build_vectorstore_for_pdf(pdf_path: str):
    """Load, chunk, embed and save vectorstore for a single PDF"""
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)
    vs_path = get_vectorstore_path(pdf_path)
    os.makedirs(vs_path, exist_ok=True)
    db.save_local(vs_path)

    return db


def load_or_build_vectorstore(pdf_path: str):
    """Load existing vectorstore or build a new one for the given PDF"""
    vs_path = get_vectorstore_path(pdf_path)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(vs_path) and os.path.exists(os.path.join(vs_path, "index.faiss")):
        return FAISS.load_local(
            vs_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        return build_vectorstore_for_pdf(pdf_path)


# ── Main Chain ─────────────────────────────────────────────────────────────────

def create_qa_chain(pdf_path: str):
    """Create a retrieval QA chain for a specific PDF"""

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in .env file")

    vectorstore = load_or_build_vectorstore(pdf_path)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful academic assistant. Answer using ONLY the provided context.
Be precise and cite relevant details from the paper.

Context:
{context}

Question:
{input}

Answer:
""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    return qa_chain


# ── Question Generation ────────────────────────────────────────────────────────

def generate_questions(qa_chain, q_type: str, count: int, topic: str = ""):
    """Generate MCQ / Short Answer / Long Answer questions from loaded paper"""

    topic_clause = f" Focus on the topic: {topic}." if topic.strip() else ""

    if q_type == "MCQ":
        prompt_text = f"""
Based on the academic paper content, generate exactly {count} multiple choice questions.{topic_clause}

IMPORTANT: Return ONLY a valid JSON array. No explanation, no markdown, no extra text.
Start your response with [ and end with ]

[
  {{
    "question": "Question text here?",
    "options": [
      {{"letter": "A", "text": "Option A text"}},
      {{"letter": "B", "text": "Option B text"}},
      {{"letter": "C", "text": "Option C text"}},
      {{"letter": "D", "text": "Option D text"}}
    ],
    "answer": "A"
  }}
]

Generate {count} such question objects. Start with [ and end with ]. Nothing else.
"""

    elif q_type == "Short Answer":
        prompt_text = f"""
Based on the academic paper content, generate exactly {count} short answer questions.{topic_clause}
Each answer should be 2-3 sentences maximum.

IMPORTANT: Return ONLY a valid JSON array. No explanation, no markdown, no extra text.
Start your response with [ and end with ]

[
  {{
    "question": "Question text here?",
    "answer": "Short answer in 2-3 sentences."
  }}
]

Generate {count} such objects. Start with [ and end with ]. Nothing else.
"""

    else:  # Long Answer
        prompt_text = f"""
Based on the academic paper content, generate exactly {count} long answer questions.{topic_clause}
Each answer should be one detailed paragraph.

IMPORTANT: Return ONLY a valid JSON array. No explanation, no markdown, no extra text.
Start your response with [ and end with ]

[
  {{
    "question": "Detailed question text here?",
    "answer": "Comprehensive paragraph answer here."
  }}
]

Generate {count} such objects. Start with [ and end with ]. Nothing else.
"""

    result = qa_chain.invoke({"input": prompt_text})
    raw = result.get("answer", "[]")

    return _parse_questions_safely(raw, q_type)


def _parse_questions_safely(raw: str, q_type: str) -> list:
    """
    Robustly parse LLM output into a list of question dicts.
    Tries 6 strategies before giving up.
    """

    # Strategy 1: Strip markdown fences and try direct parse
    clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON array with regex (handles extra text before/after)
    match = re.search(r'\[.*?\]', clean, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except json.JSONDecodeError:
            pass

    # Strategy 3: Fix common JSON issues — unescaped smart/curly quotes
    try:
        fixed = clean.replace('\u201c', '"').replace('\u201d', '"')
        fixed = fixed.replace('\u2018', "'").replace('\u2019', "'")
        parsed = json.loads(fixed)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 4: Find the outermost [ ... ] even if nested content is messy
    start = clean.find('[')
    end   = clean.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(clean[start:end + 1])
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except json.JSONDecodeError:
            pass

    # Strategy 5: Extract individual {...} objects one by one
    questions = []
    for obj_match in re.finditer(r'\{[^{}]*\}', clean, re.DOTALL):
        try:
            obj = json.loads(obj_match.group())
            if 'question' in obj:
                questions.append(obj)
        except json.JSONDecodeError:
            continue

    if questions:
        return questions

    # Strategy 6: Plain text fallback — extract Q&A pairs from raw text
    # Handles cases where model ignores JSON instruction entirely
    questions = []
    lines = raw.strip().split('\n')
    current_q = None
    current_a = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect question lines — numbered (1. / 1) ) or Q: prefix
        q_match = re.match(r'^(?:\d+[\.\)]|Q\d*[\.\):])\s*(.+)', line)
        # Detect answer lines — A: / Ans: / Answer: prefix
        a_match = re.match(r'^(?:A(?:ns(?:wer)?)?[\.\):])\s*(.+)', line, re.IGNORECASE)

        if q_match:
            if current_q and current_a:
                questions.append({
                    "question": current_q,
                    "answer": " ".join(current_a),
                    "options": []
                })
            current_q = q_match.group(1)
            current_a = []
        elif a_match and current_q:
            current_a.append(a_match.group(1))
        elif current_q and current_a:
            current_a.append(line)

    if current_q and current_a:
        questions.append({
            "question": current_q,
            "answer": " ".join(current_a),
            "options": []
        })

    if questions:
        return questions

    # All strategies failed — show raw output as a readable card
    # instead of an unhelpful error message
    return [{
        "question": f"Generated {q_type} content (tap to expand)",
        "answer": raw.strip(),
        "options": []
    }]