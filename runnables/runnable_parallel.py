from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda , RunnableSequence , RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# ---------------- LLM ----------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

prompt1  = PromptTemplate(
    template="Generate a tweet about the following topic: {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Generate a LinkedIn post about the following topic: {topic}",
    input_variables=["topic"]
)
parser = StrOutputParser()
parallel_chain = RunnableParallel({
     'tweet' : RunnableSequence(prompt1, model, parser),
    'LinkedIn': RunnableSequence(prompt2, model, parser)
}
   
)
result = parallel_chain.invoke({"topic": "Artificial Intelligence"})
print(result)