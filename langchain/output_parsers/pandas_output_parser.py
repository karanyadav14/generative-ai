# from langchain.output_parsers import PandasDataFrameOutputParser
# from langchain.prompts import PromptTemplate
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())


# llm = HuggingFaceEndpoint(
#     repo_id = 'google/gemma-2-2b-it',
#     task = 'text-generation'
# )

# model = ChatHuggingFace(llm=llm)

# parser = PandasDataFrameOutputParser()

# prompt = PromptTemplate.from_template(
#     "Give me a table of top 3 countries by GDP in {year} with columns: Country, GDP (in trillion USD). \n {format_instructions}",
#     input_variables = ['year'],
#     partial_variables = {'format_instructions':parser.get_format_instructions()}
# )

# print(prompt)
# # chain = prompt| model | parser

# # result = chain.invoke({'year':'2025'})
# # print(result)

# prompt = prompt.inovke({'year':2025})
# result = model.invoke(prompt)
# print(result.content)


from langchain.output_parsers.pandas_dataframe import PandasDataFrameOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# First, define the expected DataFrame structure
df = pd.DataFrame(columns=["Country", "GDP (in trillion USD)"])

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

# Initialize the parser with the expected DataFrame structure
parser = PandasDataFrameOutputParser(dataframe=df)

prompt = PromptTemplate(
    template="""Give me a table of top 3 countries by GDP in {year} with columns: Country, GDP (in trillion USD).
    {format_instructions}""",
    input_variables=["year"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser

try:
    result = chain.invoke({"year": "2025"})
    print(result)
except Exception as e:
    print(f"An error occurred: {e}")