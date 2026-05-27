import os
import json
import shutil
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.makedirs("docs", exist_ok=True)

TRACKER = "indexed_files.json"

def get_indexed_files():
    # Returns the set of filenames already embedded in the vector store
    if os.path.exists(TRACKER):
        with open(TRACKER) as f:
            return set(json.load(f))
    return set()

def save_indexed_files(files):
    with open(TRACKER, "w") as f:
        json.dump(list(files), f)

def pdf_to_chunks(filepath, filename):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    doc = Document(page_content=text, metadata={"source": filename})
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    return splitter.split_documents([doc])

st.set_page_config(page_title="RAG Chat", page_icon="📄")
st.title("📄 Chat with your Documents")

# ── SIDEBAR ───────────────────────────────────
st.sidebar.title("Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    indexed = get_indexed_files()
    new_files = []
    for file in uploaded_files:
        save_path = f"docs/{file.name}"
        if file.name not in indexed:
            with open(save_path, "wb") as f:
                f.write(file.read())
            new_files.append(file.name)

    if new_files:
        # Add new docs to existing store without deleting it — avoids Windows file lock
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(
            persist_directory="chroma_db_ui",
            embedding_function=embeddings
        )
        for filename in new_files:
            chunks = pdf_to_chunks(f"docs/{filename}", filename)
            vectorstore.add_documents(chunks)
            indexed.add(filename)

        save_indexed_files(indexed)
        st.sidebar.success(f"Added: {', '.join(new_files)}")
        st.cache_resource.clear()
        st.rerun()

# Show indexed documents in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Indexed documents:**")
indexed_files = get_indexed_files()
if indexed_files:
    for f in sorted(indexed_files):
        st.sidebar.caption(f"- {f}")
else:
    st.sidebar.caption("No documents yet. Upload a PDF above.")

# Rebuild button — two step to avoid Windows file lock:
# Step 1: set flag + clear cache → releases ChromaDB connection
# Step 2: on next run, see the flag → now safe to delete
if st.sidebar.button("Rebuild Index"):
    st.session_state.pending_rebuild = True
    st.cache_resource.clear()
    st.rerun()

if st.session_state.get("pending_rebuild"):
    if os.path.exists("chroma_db_ui"):
        shutil.rmtree("chroma_db_ui")
    if os.path.exists(TRACKER):
        os.remove(TRACKER)
    st.session_state.pending_rebuild = False
    st.rerun()

# ── LOAD RAG (cached) ─────────────────────────
@st.cache_resource
def load_rag():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists("chroma_db_ui"):
        vectorstore = Chroma(
            persist_directory="chroma_db_ui",
            embedding_function=embeddings
        )
        st.sidebar.info("Vector store loaded from disk")
    else:
        pdf_files = [f for f in os.listdir("docs") if f.endswith(".pdf")]
        if not pdf_files:
            return None, None

        all_chunks = []
        indexed = set()
        for filename in pdf_files:
            chunks = pdf_to_chunks(f"docs/{filename}", filename)
            all_chunks.extend(chunks)
            indexed.add(filename)

        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory="chroma_db_ui"
        )
        save_indexed_files(indexed)
        st.sidebar.info(f"Built index from {len(pdf_files)} document(s)")

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15}
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return retriever, llm

retriever, llm = load_rag()

# ── PROMPTS ───────────────────────────────────
condense_prompt = ChatPromptTemplate.from_template("""
Given the conversation history below and a new question, rewrite the new question
as a standalone question that can be understood without the history.
If the question is already standalone, return it as is.

Conversation history:
{history}

New question: {question}

Standalone question:
""")

answer_prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(messages):
    history = ""
    for m in messages[:-1]:
        role = "User" if m["role"] == "user" else "Assistant"
        history += f"{role}: {m['content']}\n"
    return history

# ── CHAT ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                for s in message["sources"]:
                    st.caption(s)

if retriever is None:
    st.info("Upload a PDF from the sidebar to get started.")
else:
    if question := st.chat_input("Ask a question about your documents..."):

        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):

                if len(st.session_state.messages) > 1:
                    history = format_history(st.session_state.messages)
                    standalone_question = (
                        condense_prompt | llm | StrOutputParser()
                    ).invoke({"history": history, "question": question})
                else:
                    standalone_question = question

                retrieved_chunks = retriever.invoke(standalone_question)
                context = format_docs(retrieved_chunks)
                answer = (
                    answer_prompt | llm | StrOutputParser()
                ).invoke({"context": context, "question": standalone_question})

            st.markdown(answer)

            sources = []
            with st.expander("Sources"):
                for i, chunk in enumerate(retrieved_chunks):
                    source = chunk.metadata["source"]
                    preview = chunk.page_content[:150].replace("\n", " ")
                    label = f"[{i+1}] {source}: \"{preview}...\""
                    st.caption(label)
                    sources.append(label)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
