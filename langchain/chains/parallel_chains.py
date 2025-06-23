from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

print(os.getenviron[""])

llm1 = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = 'text-generation'
)

model1 = ChatHuggingFace(llm=llm1)

llm2 = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = 'text-generation'
)

model2 = ChatHuggingFace(llm=llm2)



prompt1 = PromptTemplate(
    template = "Generate detailed notes on topic: {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "Generate 10 MCQ quiz on topic: {topic}",
    input_variables=['topic']
)


parser = StrOutputParser()

chain1 = prompt1 | model1 | parser
chain2 = prompt2 | model2 | parser

result1 = chain1.invoke({'topic':'XGBoost'})
result2 = chain2.invoke({'topic':'XGBoost'})

print(result1)
print(result2)