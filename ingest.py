import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Paths
DATA_DIR = "data/samples"
DB_DIR = "vectorstore"


def load_pdfs(data_dir):
    documents = []

    pdf_files = list(Path(data_dir).glob("*.pdf"))

    if not pdf_files:
        raise ValueError("❌ No PDF files found in data/samples/")

    print(f"📄 Found {len(pdf_files)} PDF files")

    for pdf in pdf_files:
        print(f"➡️ Loading: {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()
        documents.extend(docs)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"✂️ Created {len(chunks)} text chunks")

    return chunks


def create_vectorstore(chunks):
    print("🔢 Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    db.save_local(DB_DIR)

    print(f"✅ Vector store saved in: {DB_DIR}/")


def main():
    print("🚀 Starting ingestion pipeline...")

    docs = load_pdfs(DATA_DIR)

    chunks = split_documents(docs)

    create_vectorstore(chunks)

    print("🎉 Ingestion completed successfully!")


if __name__ == "__main__":
    main()
