import streamlit as st
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    prompts = df['prompt'].tolist()
    responses = df['response'].tolist()
    return prompts, responses

def embed_texts(texts, model, tokenizer):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return embeddings

def main():
    st.title("Customer Service Chatbot")

    # Initialize session state for conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Load and process data
    prompts, responses = load_and_process_data("dataset.csv")

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

    # User input
    user_input = st.text_input("You: ", "")

    if user_input:
        # Append user input to conversation
        st.session_state.conversation.append({"role": "user", "text": user_input})

        # Here you should implement a way to get a response based on the user's input
        # For simplicity, let's just echo the user input
        response = "Echo: " + user_input  # Replace with model's actual response

        # Append model response to conversation
        st.session_state.conversation.append({"role": "bot", "text": response})

    # Display conversation history
    for message in st.session_state.conversation:
        if message["role"] == "user":
            st.write(f"You: {message['text']}")
        else:
            st.write(f"Bot: {message['text']}")

if __name__ == "__main__":
    main()
