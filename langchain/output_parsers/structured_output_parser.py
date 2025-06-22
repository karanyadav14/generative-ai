from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(name='face_1', description='Fact 1 about the topic'),
    ResponseSchema(name='face_2', description='Fact 2 about the topic'),
    ResponseSchema(name='face_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "Give 3 fact about {topic} \n {format_instructions}",
    input_variables=['topic'],
    partial_variables = {'format_instructions':parser.get_format_instructions()}
)


# prompt = template.invoke({'topic':'black hole'})
# # print(prompt)
# result = model.invoke(prompt)

# final_result = parser.parse(result.content)
# print(final_result)


chain = template | model | parser

result = chain.invoke({'topic':'black hole'})
print(result)


