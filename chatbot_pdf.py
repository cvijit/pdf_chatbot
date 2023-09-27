import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import textract

# Streamlit UI
st.title("PDF Chatbot")

# Upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Load the PDF and split into pages
    loader = PyPDFLoader(uploaded_file)
    pages = loader.load_and_split()

    # Tokenize and create embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=24,
        length_function=lambda text: len(tokenizer.encode(text)),
    )

    chunks = text_splitter.create_documents(pages)

    # Create a vector database
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)

    # Create a chatbot chain
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())
    chat_history = []

    st.subheader("Chatbot")

    while True:
        user_question = st.text_input("Ask a question:")
        
        if st.button("Ask"):
            if user_question.lower() == 'exit':
                st.write("Thank you for using the PDF Chatbot!")
                break

            result = qa({"question": user_question, "chat_history": chat_history})
            chat_history.append((user_question, result['answer']))

            st.text(f"User: {user_question}")
            st.text(f"Chatbot: {result['answer']}")

# Note: You may need to adjust the import statements and some parts of the code to match your specific environment and dependencies.
