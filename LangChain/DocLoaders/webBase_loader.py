
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
import os
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)
url = "https://www.w3schools.com/html/html_basic.asp"
loader = WebBaseLoader(url)

docs = loader.load()

prompt = PromptTemplate(
    template= "Answer the following questions \n {question} from the following text \n {text}",
    input_variables=['question','text']
)
parser  = StrOutputParser()
chain = prompt | model | parser

print(chain.invoke({'question': "with which tag html paragraphs start", 'text':docs[0].page_content }))