from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
load_dotenv()
model = ChatAnthropic(model = 'claude-2', temperature=0.5,max_completetion_tokens=500)
result = model.invoke("What is the capital of France?")
print(result.content)
