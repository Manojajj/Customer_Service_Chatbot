import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

# Set the title of the app
st.title("CUSTOMER SERVICE CHATBOT 💬🤖")

# Button to create the knowledge base
if st.button("Create Knowledgebase"):
    create_vector_db()
    st.success("Knowledgebase created successfully!")

# Input field for the user to ask questions
question = st.text_input("Question: ")
if question:
    chain = get_qa_chain()
    if chain:
        response = chain(question)
        st.header("Answer")
        st.write(response["result"])
    else:
        st.error("Unable to generate a response. Please check your inputs.")
