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
import os

# ── STEP 1: LOAD ALL PDFs FROM docs/ FOLDER ───
# os.listdir() reads every file name in the folder
# We loop through, pick only .pdf files, and extract text from each one
# Each PDF becomes its own Document with its filename stored in metadata
documents = []
for filename in os.listdir("docs"):
    if filename.endswith(".pdf"):
        reader = PdfReader(f"docs/{filename}")
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        documents.append(Document(
            page_content=text,
            metadata={"source": filename}  # remember which file this came from
        ))
        print(f"Loaded: {filename} ({len(reader.pages)} pages)")

print(f"\nTotal documents loaded: {len(documents)}")

# ── STEP 2: SPLIT INTO CHUNKS ─────────────────
# chunk_size=500  → max 500 characters per chunk
# chunk_overlap=50 → shared characters between chunks so context isn't cut off
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)


chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks\n")

# ── STEP 3: EMBED + STORE IN CHROMADB ─────────
# HuggingFace embeddings run locally on your machine — no API key needed
# "all-MiniLM-L6-v2" is a small fast model (80MB), downloads once automatically
# It converts text chunks into vectors (lists of numbers)
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store vectors in ChromaDB on disk so we don't re-embed every run
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db_v3"
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

# ── STEP 7: BUILD THE RAG PIPELINE ────────────
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

    # Step A: retrieve the chunks manually so we can show sources
    retrieved_chunks = retriever.invoke(question)

    # Step B: format chunks into context string for the LLM
    context = format_docs(retrieved_chunks)

    # Step C: run the LLM with the context
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })

    print(f"\nAnswer: {answer}")

    # Step D: show which document and which text was used
    print("\nSources:")
    for i, chunk in enumerate(retrieved_chunks):
        source = chunk.metadata["source"]
        preview = chunk.page_content[:120].replace("\n", " ")
        print(f"  [{i+1}] {source} -> \"{preview}...\"")

    print("\n" + "-" * 50 + "\n")


# ── STEP 8: ASK QUESTIONS ─────────────────────
print("RAG is ready! Type 'exit' to quit.\n")

while True:
    question = input("Your question: ")
    if question.lower() == "exit":
        break

    answer = chain.invoke(question)
    print(f"\nAnswer: {answer}\n")
    print("─" * 50 + "\n")
