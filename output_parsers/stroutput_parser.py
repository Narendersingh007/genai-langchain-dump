from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder , PromptTemplate
from dotenv import load_dotenv
load_dotenv()
import os
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

# 1st Prompt - > Detailed Response
template1 = PromptTemplate(
    template = "write a detailed response about the following topic: {topic}",
    input_variables = ["topic"]
)
# 2nd Prompt -> Summarized Response
template2 = PromptTemplate(
    template = "write a 5 line summarized response about the following text : {text}",
    input_variables = ["text"]
)
prompt1 = template1.invoke({"topic": "Artificial Intelligence"})
result1 =model.invoke(prompt1)
prompt2 = template2.invoke({"text": result1.content})
result2 = model.invoke(prompt2)
print(result1.content)
print("\n\n")
print(result2.content)