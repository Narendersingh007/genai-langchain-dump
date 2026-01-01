from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()
model = ChatOpenAI(model = 'gpt-4',temperature=0.5,max_completion_tokens=500)
result = model.invoke("What is the capital of France?")
print(result.content)
