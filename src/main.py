
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

################### CHAT GPT #####################
# Langchain specific models that specify how to interact with the openAI API
chatgpt = ChatOpenAI(temperature=0)

gpt = OpenAI(temperature=0)

template = "{input}"

one_shot = PromptTemplate(template=template, input_variables=["input"])

@app.post("/one-shot")
async def one_shot_route(input: str):
    return chatgpt(one_shot.format_prompt(input=input).to_messages())

############## translator route###############
template = "You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
translator_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])

@app.post("/language-translator")
async def language_translator_route(phrase: str, input_lang: str, output_lang: str):
    print(translator_prompt.format_prompt(input_language=input_lang,
          output_language=output_lang, text=phrase).to_string())
    return chatgpt(translator_prompt.format_prompt(input_language=input_lang, output_language=output_lang, text=phrase).to_messages())

################ Few Shot Prompts ################
examples = [{
    "question": "What is interesting about the number 7?",
    "answer":
        """
Are follow up questions needed here: Yes.
Follow up: What is the square root of 7?
Intermediate answer: 2.645751311064591.
Follow up: What is the square of 7?
Intermediate answer: 49.
So the final answer is: The square root of 7 is 2.645751311064591 and the square is 49.
"""
},
{
        "question": "What is interesting about the number 8?",
        "answer":
        """
Are follow up questions needed here: Yes.
Follow up: What is the square root of 8?
Intermediate answer: 2.82842712474619.
Follow up: What is the square of 8?
Intermediate answer: 64.
So the final answer is: The square root of 8 is 2.82842712474619 and the square is 64.
"""
}]

example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}")

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)

@app.post("/few-shot")
async def few_shot_route(input: str):
    return chatgpt(few_shot_prompt.format_prompt(input=input).to_messages())

class AddInput(BaseModel):
    num1: str = Field(description="The first number to add")
    num2: str = Field(description="The second number to add")


def add(num1: str, num2: str):
    return int(num1) + int(num2)

add_tool = StructuredTool.from_function(func=add,
                                        name="Add",
                                        description="useful for adding numbers together",
                                        args_schema=AddInput)

add_agent = initialize_agent([add_tool], gpt,
                             agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


@app.post("/add")
async def add_route(phrase: str):
    return add_agent.run(phrase)

################# Data Extraction #######################


class TextDocument(BaseModel):
    number_of_words: int = Field(description="The number of words in the document")
    subject: str = Field(description="The subject of the document")
    summary: str = Field(description="A short summary of the document no more than 30 words")
    most_common_word: str = Field(description="The most common word in the document excluding stop words")

parser = PydanticOutputParser(pydantic_object=TextDocument)

extraction_prompt = PromptTemplate(
    template="Extract information from the document.\n{format_instructions}\n{input}\n",
    input_variables=["input"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

@app.post("/process-document")
async def process_document_route(doc: str):
    resp = gpt(extraction_prompt.format(input=doc))
    return parser.parse(resp)
