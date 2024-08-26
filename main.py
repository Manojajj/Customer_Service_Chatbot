import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

api_key = st.text_input("Enter Google API Key (optional)", type="password")

st.title("CUSTOMER SERVICE CHATBOT 💬🤖")

if st.button("Create Knowledgebase"):
    create_vector_db()
    st.success("Knowledgebase created successfully!")

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
