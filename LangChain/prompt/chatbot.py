from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()
import os
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)
chat_history = [
    SystemMessage(content="You are a helpful AI assistant.")
]
while True:
    user_input = input('You :')
    chat_history.append(HumanMessage(content=user_input))
    if(user_input == 'exit' or user_input == 'quit'):
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('Bot :', result.content)

print(chat_history)