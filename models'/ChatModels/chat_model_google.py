from dotenv import load_dotenv
load_dotenv()

import os
from google import genai
from langchain_core.runnables import RunnableLambda

# Initialize native Gemini client
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# Wrap Gemini call for LangChain
def gemini_llm(prompt: str) -> str:
    response = client.models.generate_content(
        model="models/gemini-flash-latest", 
        contents=prompt
    )
    return response.text

# LangChain-compatible LLM
llm = RunnableLambda(gemini_llm)

# Invoke like any LangChain model
result = llm.invoke("What is the capital of France?")
print(result)