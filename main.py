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

# Function to create the vector database
def create_vector_db():
    api_key = st.text_input("Enter Google API Key (optional)", type="password")
    
    # Initialize GooglePalm LLM if API key is provided
    llm = None
    if api_key:
        llm = GooglePalm(google_api_key=api_key, temperature=0.1)
    else:
        st.warning("Google API Key is not provided. The application may not function as expected.")
    
    # Initialize embeddings
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

    # Load the dataset
    loader = CSVLoader(file_path="dataset/dataset.csv", source_column="question")
    data = loader.load()
    
    # Create a FAISS vector database
    vectordb = FAISS.from_documents(data, instructor_embeddings)
    vectordb.save_local("faiss_index")
    st.success("Knowledgebase created successfully!")

# Function to get the QA chain
def get_qa_chain():
    # Load the vector database
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    vectordb = FAISS.load_local("faiss_index", embeddings=instructor_embeddings)
    
    # Set up the retriever
    retriever = vectordb.as_retriever(score_threshold=0.7)
    
    # Define the prompt template
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "answer" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context} 
    
    QUESTION: {question}"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Initialize the chain
    llm = GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.1)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

# Streamlit app logic
if st.button("Create Knowledgebase"):
    create_vector_db()

question = st.text_input("Question: ")
if question:
    chain = get_qa_chain()
    if chain:
        response = chain(question)
        st.header("Answer")
        st.write(response["result"])
    else:
        st.error("Unable to generate a response. Please check your inputs.")
