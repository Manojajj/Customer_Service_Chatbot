import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("distilbert/distilbert-base-uncased")

# Define the text embedding function
def embed_text(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Load and process the dataset with specified encoding
def load_and_process_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')  # Try 'utf-8' first
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')  # Fallback to 'latin1' if 'utf-8' fails
    texts = df['question'].tolist()
    answers = df['answer'].tolist()
    return texts, answers

# Create the vector database from the dataset
def create_vector_db():
    file_path = "dataset.csv"  # Path to your dataset.csv
    texts, answers = load_and_process_data(file_path)
    
    embeddings = embed_text(texts)
    vectorstore = FAISS(embeddings=embeddings, documents=texts)
    vectorstore.save_local("faiss_index")
    st.success("Knowledgebase created successfully!")

# Define the LangChain QA Chain
def get_qa_chain():
    vectorstore = FAISS.load_local("faiss_index")
    retriever = vectorstore.as_retriever()
    
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
    
    CONTEXT: {context}
    QUESTION: {question}"""

    chain = RetrievalQA.from_chain_type(
        llm=None,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    
    return chain

# Streamlit UI
st.title("Customer Service Chatbot 💬🤖")

# Create the knowledgebase (run this once or on demand)
if st.button("Create Knowledgebase"):
    create_vector_db()

question = st.text_input("Ask your question:")
if question:
    chain = get_qa_chain()
    if chain:
        response = chain(question)
        st.header("Answer")
        st.write(response["result"])
    else:
        st.error("Unable to generate a response. Please check your inputs.")
