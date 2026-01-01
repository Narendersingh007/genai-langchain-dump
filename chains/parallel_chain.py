from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder , PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os
llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)
llm2 = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template = "Generate short and simple notes from the following text: {text}",
    input_variables = ["text"]
)
prompt2 = PromptTemplate(
    template = "Generate 5 short questions from the following text: {text}",
    input_variables = ["text"]
)

prompt3 = PromptTemplate(
    template = "Merge the provided notes and questions into a single document: Notes: {notes} Questions: {quiz}",
    input_variables = ["notes", "quiz"]
)
parser = StrOutputParser()
parallel_chain = RunnableParallel(
    {
    'notes' : prompt1 | model1 | parser,
    'quiz'  : prompt2 | model2 | parser
    }
)

merge_chain = prompt3 | model1 | parser
chain = parallel_chain | merge_chain
text = """
In the study of heat transfer, Newton's law of cooling is a physical law which states that the rate of heat loss of a body is directly proportional to the difference in the temperatures between the body and its environment. The law is frequently qualified to include the condition that the temperature difference is small and the nature of heat transfer mechanism remains the same. As such, it is equivalent to a statement that the heat transfer coefficient, which mediates between heat losses and temperature differences, is a constant.

In heat conduction, Newton's law is generally followed as a consequence of Fourier's law. The thermal conductivity of most materials is only weakly dependent on temperature, so the constant heat transfer coefficient condition is generally met. In convective heat transfer, Newton's Law is followed for forced air or pumped fluid cooling, where the properties of the fluid do not vary strongly with temperature, but it is only approximately true for buoyancy-driven convection, where the velocity of the flow increases with temperature difference. In the case of heat transfer by thermal radiation, Newton's law of cooling holds only for very small temperature differences.

When stated in terms of temperature differences, Newton's law (with several further simplifying assumptions, such as a low Biot number and a temperature-independent heat capacity) results in a simple differential equation expressing temperature-difference as a function of time. The solution to that equation describes an exponential decrease of temperature-difference over time. This characteristic decay of the temperature-difference is also associated with Newton's law of cooling.

Historical background"""
result = chain.invoke({"text":text})
print(result)
chain.get_graph().print_ascii()