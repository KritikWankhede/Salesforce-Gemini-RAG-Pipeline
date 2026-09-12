from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader

app = FastAPI()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

documents = [
    Document(
        page_content="User cannot log in after password reset. Resolution: Clear browser cache and ensure cookies are enabled.",
        metadata={"title": "Password Reset Login Failure", "record_id": "CAS-1001"}
    ),
    Document(
        page_content="Printer offline error on office network. Resolution: Restart the print spooler service and reconnect to IP 192.168.1.50.",
        metadata={"title": "Office Printer Offline", "record_id": "CAS-1002"}
    )
]

# Automatically load any PDFs found in the 'data/' folder
if os.path.exists("data"):
    pdf_loader = PyPDFDirectoryLoader("data")
    pdf_docs = pdf_loader.load()
    documents.extend(pdf_docs)
    print(f"Loaded {len(pdf_docs)} PDF documents into knowledge base.")

vector_store = Chroma.from_documents(documents, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
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
    
    sources = [{"title": doc.metadata.get("title", "PDF Manual"), "id": doc.metadata.get("source", "Manual")} for doc in response["context"]]
    
    return {
        "suggested_resolution": response["answer"],
        "sources": sources
    }

# --- FEEDBACK ENDPOINT ---
class FeedbackPayload(BaseModel):
    case_id: str
    rating: str
    suggestion: str

@app.post("/api/feedback")
def receive_feedback(feedback: FeedbackPayload):
    print(f"Feedback received for Case {feedback.case_id}: {feedback.rating}")
    print(f"Suggestion was: {feedback.suggestion}")
    return {"status": "success", "message": "Feedback recorded successfully!"}

#New Case Sync Payload
class NewCasePayload(BaseModel):
    record_id: str
    title: str
    resolution: str

@app.post("/api/add-case")
def add_case_to_kb(new_case: NewCasePayload):
    try:
        # Add the newly resolved case into the running vector store
        vector_store.add_texts(
            texts=[new_case.resolution],
            metadatas=[{"title": new_case.title, "record_id": new_case.record_id}]
        )
        return {"status": "success", "message": f"Case {new_case.record_id} added to knowledge base."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
