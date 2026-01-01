from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "It enables developers to build applications that can understand and generate natural language.",
    "Delhi is the capital of India.",
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Tokyo is the capital of Japan.",
    "Canberra is the capital of Australia.",
    "Ottawa is the capital of Canada.",
    "Brasília is the capital of Brazil.",
    "Moscow is the capital of Russia."
]
query = "What is langchain?"
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

ranked = sorted(
    enumerate(scores),
    key=lambda x: x[1],
    reverse=True
)

for idx, score in ranked:
    print(f"{score:.4f} → {documents[idx]}")