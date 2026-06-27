from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# ---------------- LLM ----------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# ---------------- Classifier ----------------
classifier_prompt = PromptTemplate(
    template="""
Classify the sentiment of the following text.

Rules:
- Answer with ONLY ONE WORD
- Output must be exactly: Positive OR Negative
- Do NOT explain
- Do NOT include JSON
- Do NOT include code

Text:
{feedback}
""",
    input_variables=["feedback"],
)

classifier_chain = classifier_prompt | model | StrOutputParser()

# ---------------- Response Prompts ----------------
positive_prompt = PromptTemplate(
    template="Write a polite and appreciative response to this positive feedback:\n{feedback}",
    input_variables=["feedback"]
)

negative_prompt = PromptTemplate(
    template="Write a polite and apologetic response to this negative feedback:\n{feedback}",
    input_variables=["feedback"]
)

# ---------------- Branch ----------------
branch_chain = RunnableBranch(
    (lambda x: x.strip() == "Positive",
     positive_prompt | model | StrOutputParser()),

    (lambda x: x.strip() == "Negative",
     negative_prompt | model | StrOutputParser()),

    RunnableLambda(lambda _: "Could not classify sentiment")
)


chain = classifier_chain | branch_chain

result = chain.invoke({
    "feedback": "The product quality is excellent and I am very satisfied!"
})

print(result)
chain.get_graph().print_ascii()