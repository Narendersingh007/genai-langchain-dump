from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
documents=[
    "LangChain is a framework for developing applications powered by language models.",
    "It enables developers to build applications that can understand and generate natural language."
    "Delhi is the capital of India."
]
embedding = OpenAIEmbeddings(model='text-embedding-ada-002',dimensions=32)
result = embedding.embed_query(documents)
print(str(result))