from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder , PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import StrOutputParser
import os
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = "Generate 5 interesting facts about the following topic: {topic}",
    input_variables = ["topic"]
)

parser = StrOutputParser()

chain   = prompt | model | parser 
result = chain.invoke({"topic":"chess"})
print(result)
chain.get_graph().print_ascii()