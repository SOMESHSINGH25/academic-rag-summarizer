# 🎓 AcademIQ — Academic Research Paper Assistant

> *An intelligent RAG-powered system to query, summarize, and generate questions from academic research papers.*
---

## 📌 What is AcademIQ?

AcademIQ is a **Retrieval-Augmented Generation (RAG)** application that lets you interact intelligently with academic PDF papers. Instead of relying on an AI's pre-trained knowledge, AcademIQ forces the model to answer **only from the actual content of your uploaded paper** — making every response accurate, grounded, and traceable to specific pages.

You can:
- Ask any question about a research paper and get precise, cited answers
- Select from multiple papers without reprocessing them every time
- Auto-generate MCQ, Short Answer, or Long Answer questions for study or evaluation
- Track your question history across the session

This makes AcademIQ useful for **students, researchers, educators, and academics** who work with large volumes of technical literature.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 **Multi-PDF Support** | Load any PDF from `data/samples/` via a dropdown selector |
| 🔍 **Semantic Search** | Finds the most relevant sections of a paper using vector similarity |
| 🤖 **RAG Pipeline** | Answers are grounded strictly in the paper's content, not AI guesswork |
| 📝 **Question Generation** | Auto-generates MCQ, Short Answer, and Long Answer questions |
| 🗂️ **Per-PDF Vectorstores** | Each paper gets its own FAISS index — built once, loaded instantly after |
| 📄 **Source Page Citations** | Every answer shows which pages of the paper it was drawn from |
| 🕑 **Chat History** | Last 5 questions are saved in the sidebar for easy reference |
| 🎨 **Academic UI** | Professional academic interface built with custom CSS in Streamlit |
| 💻 **CPU Optimised** | No GPU required — runs on any standard machine |
| 🔐 **Secure API Handling** | API keys stored in `.env`, never in source code |

---

## 🏗️ Project Architecture

```
PDF File
   │
   ▼
PyPDFLoader ──► Raw Text Pages
   │
   ▼
RecursiveCharacterTextSplitter ──► ~1000 character chunks (200 overlap)
   │
   ▼
HuggingFace Embeddings (all-MiniLM-L6-v2) ──► 384-dimensional vectors
   │
   ▼
FAISS Vector Store ──► Saved to disk per PDF (vectorstore/<pdf-name>/)
        │
        │  At query time:
        ▼
User Question ──► Embed ──► Similarity Search ──► Top 4 Chunks
                                                        │
                                                        ▼
                                           Groq LLM (LLaMA 3.1 8B Instant)
                                           + Strict Context Prompt
                                                        │
                                                        ▼
                                             Answer + Source Page Numbers
```

### How Each Component Works

**PDF Loader** uses `PyPDFLoader` from LangChain Community to extract raw text from every page of the document, preserving page metadata for source citation.

**Text Splitter** uses `RecursiveCharacterTextSplitter` with a chunk size of 1000 characters and 200-character overlap. The overlap ensures important content at chunk boundaries is never lost.

**Embeddings** uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` model to convert each text chunk into a 384-dimensional vector that captures its semantic meaning — not just keywords.

**FAISS Vector Store** stores all chunk vectors in a Facebook AI Similarity Search index saved locally to disk. Each PDF gets its own dedicated index so it's only built once.

**Retriever** takes the user's question, embeds it with the same model, and performs a cosine similarity search to retrieve the 4 most relevant chunks from the paper.

**LLM (Groq + LLaMA 3.1)** receives the 4 retrieved chunks as context alongside a strict prompt instructing it to answer only from the provided context. Groq is used as the inference provider for its free tier and extremely fast response speeds.

**Question Generator** uses the same RAG chain but with a structured JSON prompt, instructing the LLM to output either MCQs (with 4 options and correct answer), Short Answers (2-3 sentences), or Long Answers (full paragraph), which are then parsed and rendered as styled cards.

---

## 📁 Project Structure

```
academic-rag-summarizer/
│
├── app.py                  # Streamlit web app (UI, tabs, session state)
├── rag_pipeline.py         # RAG logic, vectorstore management, question generation
├── ingest.py               # Standalone PDF ingestion script (optional)
├── requirements.txt        # All Python dependencies
├── .env                    # API keys — never committed to Git
│
├── data/
│   └── samples/            # Place your academic PDFs here
│
├── vectorstore/            # Auto-created — one subfolder per PDF
│   └── <pdf-name>/
│       ├── index.faiss
│       └── index.pkl
│
└── venv/                   # Virtual environment — not committed
```

---

## ⚙️ Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.12 | Core language |
| LangChain | 0.3.25 | RAG orchestration framework |
| LangChain Community | 0.3.24 | PDF loader, FAISS integration |
| LangChain Core | 0.3.63 | Prompts, chains, base abstractions |
| LangChain Groq | 0.2.3 | Groq LLM integration |
| LangChain HuggingFace | 0.1.2 | Embedding model integration |
| LangChain Text Splitters | 0.3.8 | Document chunking |
| FAISS CPU | — | Vector similarity search |
| Sentence Transformers | — | `all-MiniLM-L6-v2` embedding model |
| Groq API | — | LLaMA 3.1 8B Instant inference |
| Streamlit | — | Web UI framework |
| PyPDF | — | PDF text extraction |
| Python-dotenv | — | Environment variable management |

---

## 🛠️ Installation & Setup

### ✅ Step 1 — Clone the Repository

```bash
git clone https://github.com/SOMESHSINGH25/academic-rag-summarizer.git
cd academic-rag-summarizer
```

### ✅ Step 2 — Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### ✅ Step 3 — Install Dependencies

```bash
pip install langchain==0.3.25 langchain-community==0.3.24 langchain-core==0.3.63 langchain-huggingface==0.1.2 langchain-text-splitters==0.3.8 langchain-groq==0.2.3
pip install faiss-cpu sentence-transformers streamlit pypdf python-dotenv torch
```

> ⚠️ **Important:** Install all LangChain packages together in a single command to ensure pip resolves compatible versions. Installing them separately can cause version conflicts.

### ✅ Step 4 — Get a Groq API Key (Free)

1. Sign up at [https://console.groq.com](https://console.groq.com)
2. Go to **API Keys** and create a new key
3. Copy the key — it starts with `gsk_...`

### ✅ Step 5 — Configure Environment Variables

Create a `.env` file in the root of the project:

```
GROQ_API_KEY=gsk_your_api_key_here
```

Rules for the `.env` file:
- No quotes around the key
- No spaces around the `=`
- File named exactly `.env` (not `env.txt` or `.env.txt`)

### ✅ Step 6 — Add Your PDFs

Place your academic papers inside the samples folder:

```
data/samples/paper1.pdf
data/samples/paper2.pdf
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

> **No need to run `ingest.py` manually.** The app automatically builds the FAISS vectorstore the first time you load a PDF, and loads it instantly from disk on subsequent uses.

---

## 🖥️ How to Use

### Asking Questions

1. Select a paper from the **sidebar dropdown**
2. Click **"Load This Paper"** — wait for the success message
3. Go to the **"Ask Questions"** tab
4. Type your question and click **"Ask →"**
5. The answer appears with **source page citations** below it
6. Previous questions are saved in the sidebar and in expandable cards

Example questions:
- *What is the main contribution of this paper?*
- *What datasets were used in the experiments?*
- *What are the limitations mentioned by the authors?*
- *How does the proposed method compare to baselines?*

### Generating Questions

1. Load a paper using the sidebar
2. Go to the **"Generate Questions"** tab
3. Choose the question type: **MCQ**, **Short Answer**, or **Long Answer**
4. Select the number of questions (3, 5, 7, or 10)
5. Optionally add a topic focus (e.g. *"methodology"*, *"results"*)
6. Click **"Generate Questions →"**

MCQs are displayed with 4 options (A–D) and the correct answer highlighted in green. Short Answers show a concise 2-3 sentence response. Long Answers are shown in expandable cards with full paragraph explanations.

---

## 🌟 Key Design Decisions

**Per-PDF Vectorstores** — Rather than a single shared index, each PDF gets its own FAISS index stored under `vectorstore/<pdf-name>/`. This means switching between papers is instant after the first load, and adding new papers doesn't require re-ingesting existing ones.

**Strict Context Prompting** — The LLM prompt explicitly instructs the model to answer *only* from the provided context. This is the core of what makes RAG reliable — it prevents hallucinations and keeps answers faithful to the actual paper.

**Groq over OpenAI** — Groq provides free, fast inference for open-source models like LLaMA 3.1, making this project fully free to run with no credit card required.

**LangChain Version Pinning** — All LangChain packages are pinned to the `0.3.x` family. The `1.x` releases introduced breaking API changes. Installing packages individually with pip can silently upgrade `langchain-core` to an incompatible version, so all packages are installed together.

---

## 🔐 Security

- API keys are stored in `.env` and never committed to Git
- `.gitignore` prevents `.env` from being pushed to GitHub
- No credentials are hardcoded anywhere in source files
- The vectorstore uses `allow_dangerous_deserialization=True` only for locally built, trusted FAISS indexes

---

## 🧪 Future Improvements

- [ ] Upload PDFs directly from the UI without manual file copying
- [ ] Multi-document comparison across papers
- [ ] Automatic paper summarization on load
- [ ] Citation generation in APA / MLA format
- [ ] Fine-tuned academic domain LLM
- [ ] Cloud deployment (Streamlit Community Cloud / Hugging Face Spaces)
- [ ] Multi-language paper support
- [ ] Export questions as PDF or Word document

---

## 🙏 Acknowledgements

- **Intel Unnati & GenAI4GenZ Hackathon** — for the opportunity and motivation to build this
- **HuggingFace** — for the `all-MiniLM-L6-v2` embedding model
- **LangChain** — for the RAG orchestration framework
- **Groq** — for free and fast LLaMA inference
- **Meta AI** — for the open-source LLaMA 3.1 model
- **Facebook AI Research** — for the FAISS library

---

## 👤 Author

**Somesh Singh**

**B.Tech Information Technology**

**Bharati Vidyapeeth (Deemed To Be University), College of Engineering, Pune**

---

## 📜 License

This project is developed for educational and hackathon purposes.
