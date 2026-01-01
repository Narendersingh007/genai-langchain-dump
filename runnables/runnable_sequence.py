from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda,RunnableSequence
from dotenv import load_dotenv

load_dotenv()

# ---------------- LLM ----------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

prompt1  = PromptTemplate(
    template="Wirte a joke about the following topic: {topic}",
    input_variables=["topic"]
)   
parser = StrOutputParser()
prompt2 = PromptTemplate(
    template="Explain the following joke in a detailed manner: {text}",
    input_variables=["text"]
)



chain = RunnableSequence(prompt1, model ,parser,prompt2,model,parser)

result = chain.invoke({"topic": "programming"})
print(result)
