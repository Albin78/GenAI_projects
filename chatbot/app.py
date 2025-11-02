from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import streamlit as st
import os


load_dotenv()

## Langsmith monitoring
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond clearly for user queries"),
        ("user", "Question:{question}")
    ]
    )


## Streamlit interface
st.title("Langchain Interface with Ollama")
input_text = st.text_input("Enter the topic you want to know")


## Ollama Llama2
llm = Ollama(model='gemma')
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))