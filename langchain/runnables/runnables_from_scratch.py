import random


class NakliLLM:

    def __init__(self):
        print("LLM Created")

    def predict(self, prompt):

        response_list = [
            "Weather in Bangalore is nice",
            "There are 7 continents in the world",
            "LangChain and LangGraph are super helpful"
        ]

        return {'response':random.choice(response_list)}


# llm = NakliLLM()
# print(llm.predict("How are you?"))

class NakliPromptTemplate:

    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)


# prompt = NakliPromptTemplate(
#     template='Write {length} poem about {topic}',
#     input_variables=['length', 'topic']
# )

# print(template.format({'length':'short', 'topic':'Ajanta Caves'}))


# llm = NakliLLM()

# print(llm.predict(prompt))


class NakaliLLMChain:

    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)

        return result['response']



prompt = NakliPromptTemplate(
    template='Write {length} poem about {topic}',
    input_variables=['length', 'topic']
)

llm = NakliLLM()

chain = NakaliLLMChain(llm=llm, prompt=prompt)

response = chain.run({'length':'short', 'topic':'Western Ghat'})
print(response)