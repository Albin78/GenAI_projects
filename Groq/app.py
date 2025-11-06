import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import time

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader(web_path="https://docs.langchain.com/oss/python/langchain/overview")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
        )
    st.session_state.splitted_docs = st.session_state.text_splitter.split_documents(
        st.session_state.docs
    )
    st.session_state.vector_store = FAISS.from_documents(
        documents=st.session_state.splitted_docs, 
        embedding=st.session_state.embeddings
    )

st.title("CHAT QA USING GROQ")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
    )

prompt = ChatPromptTemplate.from_template(
    """Answer the questions based on the provided context only.
       Please provide most accurate response based on the question.
       If you don't know the answer, respond in a way that 'I don't know'.
       <context>
       {context}
       </context>
       Question: {input}

    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vector_store.as_retriever()

retrieval_chain = create_retrieval_chain(retriever, document_chain)

input_text = st.text_input("Enter your prompt here")

if input_text:
    start_time = time.process_time()
    response = retrieval_chain.invoke({"input": input_text})
    print("Response time:", time.process_time() - start_time)

    st.write(response['answer'])
