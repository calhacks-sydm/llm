from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key= os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

explain_template = """You are a helpful teaching assistant helping to understand mistakes students make. This is the problem {problem} and its solution {solution}. What is one wrong approach of solving the question that students may do"""

explain_prompt = PromptTemplate(template=explain_template, input_variables=["problem","solution"])
explain_chain=LLMChain(llm=llm,prompt=explain_prompt)
question='A store is comparing their profits throughout the year. They had profits of $1,500 in the first quarter of the year, $3,000 in the third quarter, and $2,000 in the fourth quarter. If their annual profits are $8,000, how much profit, in dollars, did they make in the second quarter?'
solution='The annual profit is $8,000.\nThe sum of the profits in the first, third, and fourth quarters is $1,500 + $3,000 + $2,000 = $6,500.\nTherefore, the profit in the second quarter is $8,000 - $6,500 = $1,500.'

def runMistakellm(problem,solution):
    explanation=explain_chain.predict(problem=problem, solution=solution)
    return(explanation)
