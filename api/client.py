import requests
import streamlit as st


def get_gemini_response(input_text: str):
    response = requests.post(url="http://localhost:8000/essay/playground/",
    json={'input': {'topic': input_text}})

    return response.json()['output']['content']


def get_ollama_response(input_text:str):
    response = requests.post(url='http://localhost:8000/essay/playground/',
    json={'input': {'topic': input_text}})

    return response.json()['output']

st.title("Langchain Demo with LLAMA2 API")
essay = st.text_input("Write an essay on")
poem = st.text_input("Write a poem on")

if essay:
    st.write(get_gemini_response(essay))

if poem:
    st.write(get_ollama_response(poem))


