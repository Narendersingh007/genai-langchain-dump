from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda,RunnableSequence,RunnableParallel,RunnablePassthrough
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


joke_gen_chain = RunnableSequence(prompt1, model ,parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation' : RunnableSequence(prompt2, model, parser)
}
)

final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)

result = final_chain.invoke({"topic": "programming"})
print(result)
