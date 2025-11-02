from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description='A simple API Server'
)

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

llm = Ollama(model="gemma")

prompt1 = ChatPromptTemplate.from_template(
    "Write me an essay about {topic} with 100 words"
)

prompt2 = ChatPromptTemplate.from_template(
    "Write me a poem about {topic} with 100 words"
)

add_routes(
    app,
    prompt1 | model,
    path="/essay"
)

add_routes(
    app,
    prompt2 | llm,
    path="/poem"
)


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)