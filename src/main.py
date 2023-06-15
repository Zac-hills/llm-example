
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

