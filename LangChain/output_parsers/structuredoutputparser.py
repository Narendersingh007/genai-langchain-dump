from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder , PromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
load_dotenv()
import os
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact_1", description="The first fact"),
    ResponseSchema(name="fact_2", description="The second fact"),
    ResponseSchema(name="fact_3", description="The third fact"),
    ResponseSchema(name="fact_4", description="The fourth fact"),
    ResponseSchema(name="fact_5", description="The fifth fact"),
]
parser = StructuredOutputParser.form_response_schemas(schema)
template = PromptTemplate(
    template = "Give me 5 facts about {topic} \n {format_instructions}",
    input_variables = ['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)
prompt = template.invoke({'topic' : "Artificial Intelligence"})
result = model.invoke(prompt)
parsed_output = parser.parse(result.content)
print(parsed_output)