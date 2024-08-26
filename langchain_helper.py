import streamlit as st
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceLLM
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

# Define the embeddings and vector store
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Example embedding model
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load the dataset and create vector database
    loader = CSVLoader(file_path="dataset.csv", source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(data, embedding_model)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    # Load the vector database and set up the QA chain
    vectordb = FAISS.load_local(vectordb_file_path, embeddings=embedding_model)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context} 
    
    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Initialize Hugging Face model for question answering
    llm = HuggingFaceLLM(model_name="distilbert-base-uncased-distilled-squad")

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain
