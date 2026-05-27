import os
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

st.set_page_config(page_title="RAG Chat", page_icon="📄")
st.title("📄 Chat with your Documents")

# ── SIDEBAR ───────────────────────────────────
st.sidebar.title("Documents")

# File uploader — accepts multiple PDFs at once
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

# When files are uploaded, save them to docs/ and trigger a rebuild
if uploaded_files:
    new_files = []
    for file in uploaded_files:
        save_path = f"docs/{file.name}"
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(file.read())
            new_files.append(file.name)

    # If any new files were saved, delete the old vector store so it rebuilds
    if new_files:
        if os.path.exists("chroma_db_ui"):
            shutil.rmtree("chroma_db_ui")
        st.sidebar.success(f"Added: {', '.join(new_files)}")
        st.cache_resource.clear()  # clear cached RAG so load_rag() runs again
        st.rerun()

# Show all current documents in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Indexed documents:**")
pdf_files = [f for f in os.listdir("docs") if f.endswith(".pdf")]
if pdf_files:
    for f in pdf_files:
        st.sidebar.caption(f"- {f}")
else:
    st.sidebar.caption("No documents yet. Upload a PDF above.")

# Rebuild button — forces re-index of all docs
if st.sidebar.button("Rebuild Index"):
    if os.path.exists("chroma_db_ui"):
        shutil.rmtree("chroma_db_ui")
    st.cache_resource.clear()
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
        documents = []
        for filename in os.listdir("docs"):
            if filename.endswith(".pdf"):
                reader = PdfReader(f"docs/{filename}")
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                documents.append(Document(
                    page_content=text,
                    metadata={"source": filename}
                ))

        if not documents:
            return None, None

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
        chunks = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db_ui"
        )
        st.sidebar.info(f"Built index from {len(documents)} document(s)")

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
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

# ── CHAT HISTORY ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                for s in message["sources"]:
                    st.caption(s)

# ── CHAT INPUT ────────────────────────────────
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
