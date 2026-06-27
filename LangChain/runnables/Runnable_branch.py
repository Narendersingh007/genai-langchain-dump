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
prompt1 = PromptTemplate(
    template="Wirte a detailed report about the following topic: {topic}",
    input_variables=["topic"]
)
parser = StrOutputParser()
prompt2 = PromptTemplate(
    template="Summarize the following text in 5 lines: {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split())>200 , RunnableSequence(
        prompt2,
        model,
        parser
    )),
    RunnablePassthrough()
)
final_chain = RunnableSequence(
    report_gen_chain,
    branch_chain
)
result = final_chain.invoke({"topic": "Climate Change"})
print(result)
