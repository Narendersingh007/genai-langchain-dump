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
prompt1 = PromptTemplate(
    template = "Generate a detailed response about the following topic: {topic}",
    input_variables = ["topic"]
)
prompt2 = PromptTemplate(
    template = "Generate a 5 pointer summary about the following text : {text}",
    input_variables = ["text"]
)
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser 

result = chain.invoke({"topic":"stephen hawking"})
print(result)
chain.get_graph().print_ascii()