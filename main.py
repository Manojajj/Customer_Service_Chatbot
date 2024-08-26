import streamlit as st
from transformers import AutoTokenizer, AutoModelForMaskedLM
from langchain import LangChain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import numpy as np

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

# Set up the LangChain vector store
def create_vector_db():
    # Load your dataset
    dataset = [
        {"text": "What is your return policy?", "answer": "You can return items within 30 days."},
        {"text": "How do I track my order?", "answer": "You can track your order using the tracking link in your email."}
        # Add more examples as needed
    ]
    
    texts = [item["text"] for item in dataset]
    answers = [item["answer"] for item in dataset]
    
    embeddings = embed_text(texts)
    vectorstore = FAISS(embeddings=embeddings, documents=texts)
    vectorstore.save_local("faiss_index")

# Define the LangChain QA Chain
def get_qa_chain():
    # Load the vector store
    vectorstore = FAISS.load_local("faiss_index")
    
    # Set up the retriever
    retriever = vectorstore.as_retriever()
    
    # Define the prompt template
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
    
    CONTEXT: {context}
    QUESTION: {question}"""

    chain = RetrievalQA.from_chain_type(
        llm=None,  # No LLM here; we are using a simple retriever
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    
    return chain

# Streamlit UI
st.title("Customer Service Chatbot 💬🤖")

if st.button("Create Knowledgebase"):
    create_vector_db()
    st.success("Knowledgebase created successfully!")

question = st.text_input("Ask your question:")
if question:
    chain = get_qa_chain()
    if chain:
        response = chain(question)
        st.header("Answer")
        st.write(response["result"])
    else:
        st.error("Unable to generate a response. Please check your inputs.")
