from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

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

prompt3 = PromptTemplate(
    template="Merge provided notes and quiz into single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables = ['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser


chain = parallel_chain | merge_chain

result = chain.invoke({'topic':'XGBoost'})

print("Response: ", result)

chain.get_graph().print_ascii()