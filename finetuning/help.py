import json
from wrong import runMistakellm
from guidance import runGuidancellm
with open('miniMath.jsonl', 'r') as json_file:
    json_list = list(json_file)
dataList=[]
count=0
for json_str in json_list:
    if count >5:
        count=0
        jj=open("math_new.json","w")
        json.dump(dataList,jj)
        jj.close()
    data={}
    result = json.loads(json_str)
    #print(f"result: {result}")
    problemSoln=result['Text']
    psList= problemSoln.split(" [Response] ")
    problem= psList[0]
    solution= psList[1]
    data["problem"]= problem.split("[Math] ")[1]
    print(data["problem"])
    data["solution"]=solution
    data["wrong_approach"]=runMistakellm(problem,solution)
    data["guidance"]= runGuidancellm(problem,data["wrong_approach"])
    res= json.dumps(data)
    dataList.append(res)
    count+=1
print(dataList)