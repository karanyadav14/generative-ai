from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template = "Generate in depth analysis on {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "Give 5 point summary of following text \n {text}",
    input_variables = ['text']

)

parser = StrOutputParser()


chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Generative AI'})
print(result)