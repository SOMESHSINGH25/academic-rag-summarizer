# 📄 Academic Paper Summarizer using RAG

An AI-powered tool that summarizes and answers questions from academic research papers using **Retrieval-Augmented Generation (RAG)**.

This project is built for the **GenAI4GenZ Hackathon by Intel Unnati** and focuses on handling long academic documents efficiently while preserving technical accuracy.

---

## 🚀 Features

- 📂 Upload and process academic PDF papers  
- 🔍 Semantic search using vector embeddings  
- 🤖 Retrieval-Augmented Generation (RAG) pipeline  
- 🧠 Accurate technical summarization  
- 🌐 Interactive web interface using Streamlit  
- 💻 Optimized for CPU-based systems  
- 🔐 Secure API key handling using environment variables  

---

## 🏗️ Project Architecture

PDF → Text Split → Embeddings → FAISS → Retriever → LLM → Summary


### 🔧 Components

- **PDF Loader** – Extracts text from research papers  
- **Text Splitter** – Divides text into manageable chunks  
- **Embeddings** – Converts text into vectors  
- **FAISS** – Stores vectors for fast retrieval  
- **LLM** – Generates accurate summaries  
- **Streamlit UI** – Provides user interface  

---

## 📁 Project Structure

academic-rag-summarizer/
│
├── app.py # Streamlit web app
├── ingest.py # PDF ingestion and vector creation
├── rag_pipeline.py # RAG logic
├── requirements.txt # Dependencies
├── .env # API key (ignored in Git)
│
├── data/
│ └── samples/ # Academic PDFs
│
└── venv/ # Virtual environment (not committed)


---

## ⚙️ Technologies Used

- Python 3.12  
- LangChain  
- FAISS (CPU)  
- Sentence-Transformers  
- HuggingFace Transformers  
- Streamlit  
- PyPDF  
- Torch  
- Python-dotenv  

---

## 🛠️ Installation & Setup

### ✅ **1️⃣ Clone the Repository**

git clone https://github.com/SOMESHSINGH25/academic-rag-summarizer.git
cd academic-rag-summarizer

##     ✅ 2️⃣ Create Virtual Environment

python -m venv venv
venv\Scripts\activate

##     ✅ 3️⃣ Install Dependencies

pip install -r requirements.txt
Or install manually:

pip install langchain faiss-cpu sentence-transformers transformers streamlit pypdf python-dotenv

##  ✅ 4️⃣ Setup Environment Variables

Create a .env file in the root directory:

OPENAI_API_KEY=your_api_key_here
##  ✅ 5️⃣ Add Sample PDFs

Place academic papers inside:

data/samples/
Example:

data/samples/paper1.pdf
▶️ Usage
✅ Step 1: Build Vector Database
python ingest.py
This will process PDFs and create embeddings.

✅ Step 2: Run Application
streamlit run app.py
✅ Step 3: Ask Questions
Enter your query

Click "Generate Summary"

Get AI-powered response

🌟 Creative / Unique Feature
📌 Intelligent Academic Compression
This project introduces an optimized compression-based RAG pipeline that:

✅ Reduces redundant content

✅ Preserves technical terminology

✅ Improves response speed

✅ Enhances contextual relevance

##   Additional Enhancements

Chunk overlap strategy improves citation continuity

Lightweight embedding model for CPU efficiency

Modular design for easy extension

📊 Performance Optimization

Uses FAISS CPU for fast similarity search

MiniLM embedding model for low resource usage

Efficient chunking strategy

No GPU dependency

## 🔐 Security

API keys stored using .env

.gitignore prevents sensitive data leaks

No credentials in source code

##  🧪 Future Improvements

Multi-document comparison

Citation generation

PDF upload via UI

Fine-tuned academic LLM

Cloud deployment

Multi-language support

##   📜 License
This project is developed for educational and hackathon purposes.

##  👤 Author

**Somesh Singh**
**B.Tech Information Technology**
**Bharati Vidyapeeth (Deemed To Be University), College of Engineering, Pune**

##  🙏 Acknowledgements

Intel GenAI4GenZ Hackathon Team

HuggingFace Community

LangChain Developers

Open Source Contributors

