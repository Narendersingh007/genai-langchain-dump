from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate.from_messages([
    ('system', "You are a helpful {domain} expert."),
    ('human', "Please explain in simple terms the concept of {topic}")
])
    
prompt = chat_template.format_prompt(
    
    domain="machine learning",
    topic="transformers"
)
print(prompt)