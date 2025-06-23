from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

from langchain.schema.runnable import RunnableBranch, RunnableLambda


from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()


class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description = 'Give sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)



prompt = PromptTemplate(
    template = "Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instructions}",
    input_variables = ['feedback'],
    partial_variables = {'format_instructions':parser2.get_format_instructions()}
)

classifier_chain = prompt | model | parser2

# print(classifier_chain.invoke({'feedback':'This is wonderful smartphone'}).sentiment)


prompt2 = PromptTemplate(
    template = "Write an appropriate response for this positive feedback {feedback}",
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = "Write an appropriate response for this negative feedback {feedback}",
    input_variables = ['feedback']
)


branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive', prompt2|model|parser),
    (lambda x:x.sentiment=='negative', prompt3|model|parser),
    RunnableLambda(lambda x: "Could not find the sentiment")
)


chain = classifier_chain | branch_chain


result = chain.invoke({'feedback':'This is wonderful smartphone'})
print(result)