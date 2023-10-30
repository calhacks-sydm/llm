import json
from categorise import runCategorisellm
from simplify import runSimplifyllm
from explainer import runExplainerllm

f = open('data/2024.json')
new_data_list=[]
data = json.load(f)
for question in data:
    print(question["question_id"])
    problem= question["question_description"]
    answer= question["answer"]
    topic= runCategorisellm(problem)
    question["topic"]=topic
    simplifyDict= runSimplifyllm(problem,answer)
    hint= simplifyDict["hint"]
    question["hint"]= hint
    sp1= simplifyDict["sub_problem 1"]
    sp2= simplifyDict["sub_problem 2"]
    subproblemList=[]
    subproblemList.append(sp1)
    subproblemList.append(sp2)
    question["subproblem"]= subproblemList
    explained= runExplainerllm(problem,answer)
    question["simplifiedAns"]= explained
    jj=open("data/2024_new.json","w")
    json.dump(data,jj)
    jj.close()
f.close()
