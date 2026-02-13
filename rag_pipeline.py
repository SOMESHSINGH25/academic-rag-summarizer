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

Return ONLY a valid JSON array with this exact structure:
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

Generate {count} such question objects. Return ONLY the JSON array, no other text.
"""

    elif q_type == "Short Answer":
        prompt_text = f"""
Based on the academic paper content, generate exactly {count} short answer questions.{topic_clause}
Each answer should be 2-3 sentences.

Return ONLY a valid JSON array with this exact structure:
[
  {{
    "question": "Question text here?",
    "answer": "Short answer here in 2-3 sentences."
  }}
]

Generate {count} such objects. Return ONLY the JSON array, no other text.
"""

    else:  # Long Answer
        prompt_text = f"""
Based on the academic paper content, generate exactly {count} detailed long answer questions.{topic_clause}
Each answer should be a thorough paragraph.

Return ONLY a valid JSON array with this exact structure:
[
  {{
    "question": "Detailed question text here?",
    "answer": "Comprehensive answer here in a full paragraph."
  }}
]

Generate {count} such objects. Return ONLY the JSON array, no other text.
"""

    result = qa_chain.invoke({"input": prompt_text})
    raw = result.get("answer", "[]")

    # Parse JSON safely
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        questions = json.loads(clean)
        return questions
    except json.JSONDecodeError:
        # Fallback: try to extract JSON array from text
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return [{"question": "Could not parse questions. Try again.", "answer": raw, "options": []}]