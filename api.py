import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastembed import TextEmbedding

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FastEmbedWrapper(Embeddings):
    def __init__(self):
        self.model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return [list(v) for v in self.model.embed(texts)]

    def embed_query(self, text):
        return list(list(self.model.embed([text]))[0])

print("Loading vector store...")
embeddings = FastEmbedWrapper()
vectorstore = Chroma(
    persist_directory="chroma_db_prod",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 15}
)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

answer_prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print("API ready.")

@app.get("/")
def health_check():
    return {"status": "RAG API is running"}

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: AskRequest):
    chunks = retriever.invoke(request.question)
    context = format_docs(chunks)
    answer = (
        answer_prompt | llm | StrOutputParser()
    ).invoke({"context": context, "question": request.question})
    sources = list(set(chunk.metadata["source"] for chunk in chunks))
    return {"answer": answer, "sources": sources}
