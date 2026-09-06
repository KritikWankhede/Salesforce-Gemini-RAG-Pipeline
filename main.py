from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# Load Gemini and ChromaDB
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
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
