from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "Hello world"
result = embedding.embed_query(text)

print(len(result))
print(result[:10])  # first 10 values