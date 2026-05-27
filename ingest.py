import os
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

DOCS_FOLDER = "docs"
CHROMA_DIR = "chroma_db_prod"

def ingest():
    print("Loading PDFs...")
    documents = []
    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".pdf"):
            reader = PdfReader(f"{DOCS_FOLDER}/{filename}")
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append(Document(
                page_content=text,
                metadata={"source": filename}
            ))
            print(f"  Loaded: {filename} ({len(reader.pages)} pages)")

    if not documents:
        print("No PDFs found in docs/ folder.")
        return

    print(f"\nSplitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = splitter.split_documents(documents)
    print(f"  {len(chunks)} chunks created")

    print("\nEmbedding and saving to ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"\nDone. Vector store saved to '{CHROMA_DIR}'")
    print(f"Total chunks indexed: {len(chunks)}")

if __name__ == "__main__":
    ingest()
