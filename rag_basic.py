from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader

# ── STEP 1: LOAD PDF ──────────────────────────
# PdfReader opens the PDF and reads each page separately
# We extract the text from every page and join it into one string
reader = PdfReader("document.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# Wrap in a Document object (LangChain's standard format: text + metadata)
# metadata "source" tells us where the text came from — useful later when showing sources
documents = [Document(page_content=text, metadata={"source": "document.pdf"})]
print(f"Loaded {len(reader.pages)} pages from PDF")

# ── STEP 2: SPLIT INTO CHUNKS ─────────────────
# chunk_size=500  → max 500 characters per chunk
# chunk_overlap=50 → shared characters between chunks so context isn't cut off
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks\n")

# ── STEP 3: EMBED + STORE IN CHROMADB ─────────
# HuggingFace embeddings run locally on your machine — no API key needed
# "all-MiniLM-L6-v2" is a small fast model (80MB), downloads once automatically
# It converts text chunks into vectors (lists of numbers)
print("Loading embedding model (downloads once ~80MB)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store vectors in ChromaDB on disk so we don't re-embed every run
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
print("Vector store ready!\n")

# ── STEP 4: RETRIEVER ─────────────────────────
# Searches ChromaDB and returns the top 3 most relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── STEP 5: PROMPT TEMPLATE ───────────────────
# Instructions we give Gemini — {context} and {question} are filled in at runtime
# "only the context below" stops Gemini from inventing answers
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
""")

# ── STEP 6: LLM ───────────────────────────────
# Gemini is only used here — for reading the chunks and writing the answer
# temperature=0 → consistent, factual answers
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ── STEP 7: BUILD THE CHAIN (RAG PIPELINE) ────
# The | pipe connects steps like an assembly line:
# question → retriever finds chunks → format_docs joins them →
# prompt fills in template → llm generates answer → StrOutputParser extracts text

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── STEP 8: ASK QUESTIONS ─────────────────────
print("RAG is ready! Type 'exit' to quit.\n")

while True:
    question = input("Your question: ")
    if question.lower() == "exit":
        break

    answer = chain.invoke(question)
    print(f"\nAnswer: {answer}\n")
    print("─" * 50 + "\n")
