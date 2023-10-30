import together
import os
from dotenv import load_dotenv
load_dotenv()

base_model_name = "togethercomputer/llama-2-7b-chat"
together.Models.start(base_model_name)
def format_to_llama2_chat(system_prompt, user_model_chat_list):
    growing_prompt = f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>>"""
    for user_msg, model_answer in user_model_chat_list:
        growing_prompt += f""" {user_msg} [/INST] {model_answer} </s>"""

    return growing_prompt
question= "(4 points) If T(n) = T(n−1)+2 ·T(n−2) (We have T(1) = 1 and T(2) = 2), then T(n) = O(?) Note: In case you need to use the quadratic formula, the solutions to the equation ax2+bx+c = 0 are x = −b ± pb2 − 4ac 2a"
answer="We can lower bound this with T(n)≥3T(n−2). This yields T(n) = Ω(3n/2)≈1.732n. We can upper bound this with T(n)≤3T(n−1). This yields T(n) =O(3n). Therefore, we can guess that it is an exponential. So, T(n) =anfor some a∈[1.732,3]. Plugging this back in, we get an=an−1+ 2an−2, ora2−a−2 = 0. Solving this using the quadratic formula yields a=−1,2 Since we know√ 3 is a lower bound on the value of a, we must have a= 2, and we get that T(n) = Θ (2n)."
userInput="For each T(n), there are 2 sub problems and each sub-problem reduces the problem size by a constant amount. These subproblems will also result in 2 more sub-problems each, causing the structure to resemble a tree. The runtime of T(n)=O(nlogn) if we consider the multiplication and addition to occur in O(1). This can however be improved with dynamic programming. If each T(n) is only computed once and memorized, we can retrieve results in O(1). Since there are n-1 numbers to calculate, the recurrence is O(n)"

test_chat_prompt="<s>[INST] <<SYS>> You are a friendly and helpful teaching assistant <</SYS>> You have posed this question "+ question +" and the right answer to the question is "+ answer +". Your student has submitted his attempt as"+ userInput +". Analyse his attempt with respect to the provided question and answer and check if he is correct. Point out where he has gone right and if he has errors, point out those out as well. [/INST]"
output = together.Complete.create(
  prompt = test_chat_prompt,
  model = base_model_name,
  max_tokens = 256,
  temperature = 0.6,
  top_k = 90,
  top_p = 0.8,
  repetition_penalty = 1.1,
  stop = ['</s>', '[/INST]']
)
print(output)
together.Models.stop(base_model_name)