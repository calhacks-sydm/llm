from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key= os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

guidance_template = """You are a helpful teaching assistant helping to point out mistakes student make and explain the mistake in an understandable manner. This is the problem {problem} and the student's approach is {wrongApproach}"""

guidance_prompt = PromptTemplate(template=guidance_template, input_variables=["problem","wrongApproach"])
guidance_chain=LLMChain(llm=llm,prompt=guidance_prompt)
problem='A store is comparing their profits throughout the year. They had profits of $1,500 in the first quarter of the year, $3,000 in the third quarter, and $2,000 in the fourth quarter. If their annual profits are $8,000, how much profit, in dollars, did they make in the second quarter?'
solution='The annual profit is $8,000.\nThe sum of the profits in the first, third, and fourth quarters is $1,500 + $3,000 + $2,000 = $6,500.\nTherefore, the profit in the second quarter is $8,000 - $6,500 = $1,500.'
wrongApproach='One wrong approach that students may do is to assume that the profits in each quarter are equal. They may divide the annual profit of $8,000 by 4 (the number of quarters) and conclude that the profit in the second quarter is $2,000. However, this assumption is incorrect because the profits in each quarter are not necessarily equal.'
def runGuidancellm(problem, wrongApproach):
    explanation=guidance_chain.predict(problem=problem, wrongApproach=wrongApproach)
    return(explanation)
