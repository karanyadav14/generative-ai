from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

prompt = PromptTemplate(
    template = "Give top 5 most asked interview questions on topic: {topic}",
    input_variables = ['topic']
)


llm = HuggingFaceEndpoint(
    repo_id ="meta-llama/Meta-Llama-3-8B-Instruct",
    task = 'text-generation'
)
model = ChatHuggingFace(llm = llm)


parser = StrOutputParser()

chain = prompt | model | parser


result = chain.invoke({"topic":"Python"})

print(result)

print("-"*50)
chain.get_graph().print_ascii()