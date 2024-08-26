import streamlit as st
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit app title
st.title("CUSTOMER SERVICE CHATBOT 💬🤖")

# Get the API key from Streamlit's text input
api_key = st.text_input("Enter Google API Key (optional)", type="password")

# Initialize the GooglePalm LLM
llm = None
if api_key:
    try:
        llm = GooglePalm(google_api_key=api_key, temperature=0.1)
    except ImportError as e:
        st.error(f"Failed to initialize GooglePalm: {e}")
else:
    st.warning("Google API Key is not provided. The application may not function as expected.")

# Define the embeddings
instructor_embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")

vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path="dataset/dataset.csv", source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(data, instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embeddings=instructor_embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context} 
    
    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = None
    if llm:
        try:
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                input_key="query",
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
        except Exception as e:
            st.error(f"Failed to create QA chain: {e}")
    else:
        st.error("LLM is not initialized. Please provide a valid Google API Key.")

    return chain

# Streamlit UI for creating the knowledge base
if st.button("Create Knowledgebase"):
    create_vector_db()
    st.success("Knowledgebase created successfully!")

# Streamlit UI for asking questions
question = st.text_input("Question: ")
if question:
    chain = get_qa_chain()
    if chain:
        try:
            response = chain(question)
            st.header("Answer")
            st.write(response["result"])
        except Exception as e:
            st.error(f"Error in generating response: {e}")
    else:
        st.error("Unable to generate a response. Please check your inputs.")
