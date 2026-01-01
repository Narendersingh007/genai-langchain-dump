from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os
llm = HuggingFacePipeline.from_model_id(
    model_id='openai-community/gpt2',
    task ='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=500
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of France?")
print(result.content)