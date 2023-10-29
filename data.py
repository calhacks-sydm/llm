import json
from categorise import runCategorisellm
from simplify import runSimplifyllm
from explainer import runExplainerllm
#PROBLEMS- 2016,2018
#2017 dates wrong
f = open('data/2015.json')
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
   # sp3= simplifyDict["sub_problem 3"]
    subproblemList=[]
    subproblemList.append(sp1)
    subproblemList.append(sp2)
    #subproblemList.append(sp3)
    question["subproblem"]= subproblemList
    explained= runExplainerllm(problem,answer)
    question["simplifiedAns"]= explained
    jj=open("data/2015_new.json","w")
    json.dump(data,jj)
    jj.close()
f.close()
