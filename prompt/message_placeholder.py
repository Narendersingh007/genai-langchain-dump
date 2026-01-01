from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

# chat template 
chat_template = ChatPromptTemplate.from_messages([
    ('system', "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="history"),

    ('human', "{query}")
])
chat_history = []
with open('chat_history.txt', 'r') as file:
    chat_history = file.readlines()

print(chat_history)

# create prompt
prompt = chat_template.invoke({'history':chat_history, 'query':'Where is my refund'})

print(prompt)