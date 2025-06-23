from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)


class Person(BaseModel):
    name : str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description = "Name of the city the person belongs to")

parser = PydanticOutputParser(pydantic_object =Person)

template = PromptTemplate(
    template = "Generate the name, age and city of fictional {place} person \n {format_instruction}",
    input_variables = ['place'],
    partial_variables = {'format_instruction':parser.get_format_instructions()}

)


# prompt = template.invoke({'place':'Indian'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)
# print(final_result)

chain = template | model | parser

result = chain.invoke({'place':'UK'})

print(result)


