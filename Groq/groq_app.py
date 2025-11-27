import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

st.title("CHATGROQ With LLAMA3")

llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Provide only the accurate answer. If no correct response is there
    reply as "I am not sure about the question." 
    <context>
    {context}
    </context>
    Question: {query}
    
    """
)


def vector_embeddings():

    if "vectors" not in st.session_state:

        st.session_state.embeddings = SentenceTransformerEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader(path=r"C:\Users\hp\OneDrive\เอกสาร\GenAI\Groq\ai_articles")
        st.session_state.docs = st.session_state.loader.load()

        if not st.session_state.docs:
            st.error("No documents found at the given path")
            return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])

        if not st.session_state.final_docs:
            st.error("Text splitting failed. PDFs might be empty")
            return

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
    


prompt1 = st.text_input("Enter you question from the documents")

if st.button("Documents Embedding"):
    vector_embeddings()
    st.write("Vector Store DB is ready")



if prompt1:
    start = time.process_time()
    retriever = st.session_state.vectors.as_retriever()
    chain = ({"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        )
        
    response = chain.invoke(prompt1)
    print("Response Time:", time.process_time() - start)
    st.write(response.content)


