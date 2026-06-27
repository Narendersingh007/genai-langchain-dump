from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder , PromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()
import os
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()
template  = PromptTemplate(
    template = "Give me 5 facts about {topic} \n {format_instructions}",
    input_variables = ['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)
chain = template | model | RunnableLambda(lambda x: x.content) | parser
final_result = chain.invoke({'topic' : "Artificial Intelligence"})
print(final_result)
print(type(final_result))