import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
import streamlit as st
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# Load and process the dataset
def load_and_process_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')  # Default to UTF-8 encoding
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Try an alternative encoding
    
    prompts = df['prompt'].tolist()
    responses = df['response'].tolist()
    
    return prompts, responses

# Create the vector database
def create_vector_db(prompts, answers):
    embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")
    vector_store = FAISS.from_texts(prompts, embeddings)
    return vector_store, answers

# Load the model
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
    nlp_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=nlp_pipeline)
    return llm

# Main function to run the app
def main():
    st.title("Customer Service Chatbot")
    
    # Load the dataset
    prompts, answers = load_and_process_data("dataset.csv")
    
    # Create the vector database
    vector_store, answers = create_vector_db(prompts, answers)
    
    # Load the model
    llm = load_model()
    
    # Get user query
    user_query = st.text_input("Ask a question:")
    
    if user_query:
        # Perform similarity search
        docs = vector_store.similarity_search(user_query)
        if docs:
            # Get the most relevant answer
            context = docs[0].page_content
            response = llm(f"{context} {user_query}")
            st.write("Answer:", response["text"])
        else:
            st.write("No relevant answer found.")

if __name__ == "__main__":
    main()
