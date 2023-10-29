from langchain import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key= os.getenv('OPENAI_API_KEY')

#COMMENTS

response_schemas = [
    ResponseSchema(name="isCorrect", description="Boolean of whether user's attempt is right"),
    ResponseSchema(name="Comments", description="Areas of user's attempt where he has made mistakes"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="You are a friendly and helpful teaching assistant. You have posed this question{question} and the right answer to the question is {answer}. Your student has submitted his attempt as {user_input}. Analyse his attempt with respect to the provided question and answer and check if he is correct. Point out where he has gone right and if he has errors, point out those out as well. {format_instructions}",
    input_variables=["question", "answer","user_input"],
    partial_variables={"format_instructions": format_instructions}
)

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt
)
question= "(4 points) If T(n) = T(n−1)+2 ·T(n−2) (We have T(1) = 1 and T(2) = 2), then T(n) = O(?) Note: In case you need to use the quadratic formula, the solutions to the equation ax2+bx+c = 0 are x = −b ± pb2 − 4ac 2a"
answer="We can lower bound this with T(n)≥3T(n−2). This yields T(n) = Ω(3n/2)≈1.732n. We can upper bound this with T(n)≤3T(n−1). This yields T(n) =O(3n). Therefore, we can guess that it is an exponential. So, T(n) =anfor some a∈[1.732,3]. Plugging this back in, we get an=an−1+ 2an−2, ora2−a−2 = 0. Solving this using the quadratic formula yields a=−1,2 Since we know√ 3 is a lower bound on the value of a, we must have a= 2, and we get that T(n) = Θ (2n)."
attempt="For each T(n), there are 2 sub problems and each sub-problem reduces the problem size by a constant amount. These subproblems will also result in 2 more sub-problems each, causing the structure to resemble a tree. The runtime of T(n)=O(nlogn) if we consider the multiplication and addition to occur in O(1). This can however be improved with dynamic programming. If each T(n) is only computed once and memorized, we can retrieve results in O(1). Since there are n-1 numbers to calculate, the recurrence is O(n)"
output =llm_chain.predict(question=question, answer=answer, user_input=attempt)
print(output_parser.parse(output))

#SAMPLE OUTPUT
# {'isCorrect': 'false', 'Comments': "The student's attempt is incorrect. Here are the mistakes:\n\n1. The student incorrectly assumes that each T(n) has 2 subproblems. In reality, T(n) = T(n−1) + 2·T(n−2) means that T(n) has 2 subproblems for T(n−2) and 1 subproblem for T(n−1).\n\n2. The student incorrectly assumes that the runtime of T(n) is O(nlogn). The correct runtime is O(2^n) as shown in the provided answer.\n\n3. The student suggests using dynamic programming to improve the runtime, but this is not applicable in this case as the recurrence relation does not have overlapping subproblems.\n\n4. The student's analysis of O(n) for the recurrence is incorrect. The correct runtime is O(2^n) as shown in the provided answer."}