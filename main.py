from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

app = FastAPI()

# 1. Initialize Gemini and an in-memory/ephemeral Chroma database with seed data
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# We seed initial historical cases directly in code so Render never depends on a missing folder
seed_documents = [
    Document(
        page_content="User cannot log in after password reset. Resolution: Clear browser cache and ensure cookies are enabled.",
        metadata={"title": "Password Reset Login Failure", "record_id": "CAS-1001"}
    ),
    Document(
        page_content="Printer offline error on office network. Resolution: Restart the print spooler service and reconnect to IP 192.168.1.50.",
        metadata={"title": "Office Printer Offline", "record_id": "CAS-1002"}
    )
]

# Create vector store in memory or transient storage
vector_store = Chroma.from_documents(seed_documents, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

class CaseQuery(BaseModel):
    subject: str
    description: str

@app.post("/api/resolve-case")
def resolve_case(case: CaseQuery):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Salesforce assistant. Use the provided context below to suggest a resolution.\n\nContext:\n{context}"),
        ("human", "Subject: {subject}\nDescription: {description}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({
        "input": f"{case.subject} {case.description}", 
        "subject": case.subject, 
        "description": case.description
    })
    
    sources = [{"title": doc.metadata.get("title"), "id": doc.metadata.get("record_id")} for doc in response["context"]]
    
    return {
        "suggested_resolution": response["answer"],
        "sources": sources
    }
