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
def word_count(text) :
    return len(text.split())
prompt  = PromptTemplate(
    template="Wirte a joke about the following topic: {topic}",
    input_variables=["topic"]
)   
parser = StrOutputParser()
joke_gen_chain = RunnableSequence(prompt, model,parser)
parallel_chain = RunnableParallel({
    'joke_chain' : RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})
final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)

result = final_chain.invoke({"topic": "computers"})
print(result)