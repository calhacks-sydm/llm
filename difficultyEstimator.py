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
#sequential chain not possible due to the processing of results from reduce chain to rank chain

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

reduce_template = """The following is an examination question, which is delimeted by ```. Summarise the challenges of the problem and the approach taken to solve it.
For example,
For question "Wilson’s Werewolves  nfriends are playing a game. k of them are werewolves, while the other n−kof them are villagers. The game is set in a cave with some number of rooms, and rooms may be connected by corridors. At the start of the game, there is a set of broken rooms in the cave that require repairs, and each of the players spawns in a random room. There is no limit on the number of players that can spawn ineach room, and it is possible for a player to spawn in a broken room. Everyone then rushes to theclosest broken room (determined by number of corridors to get to the room) to either help fix the cave (villager), or pretend to help fix the cave (werewolf). We want to determine how many villagers are in danger because they end up in the same broken room as a werewolf. Assume that the cave can be represented as an unweighted, undirected graph G= (V, E), where the vertices represent rooms and the edges represent corridors connecting rooms. The set of broken rooms is B⊆V. Further assume that all the players take the shortest path to the closest broken room, and there are no ties. Finally, assume that every room has the capacity to hold any number of players. Design an efficient algorithm to solve this problem"

The summarised challenge is "The question presents a multifaceted challenge involving a game set in a cave system, where players, categorized as villagers or werewolves, interact based on defined rules. The intricacy arises primarily from the intertwining elements: random player spawns, cave structure represented by an undirected graph, movement based on shortest paths to broken rooms, and the objective to ascertain villagers at risk. Furthermore, the solution requires a grasp of graph theory concepts, particularly BFS and DFS algorithms. The combination of understanding and integrating these diverse elements, along with the task of designing an efficient algorithm that adheres to the given constraints"

and its summarised approach is "Creating an auxillary node, run BFS from source node and traverse tree using DFS"
{format_instructions}
```
{question}
```
"""

response_schemas_reduce = [
    ResponseSchema(name="challenge", description="Challenges of the given problem"),
    ResponseSchema(name="approach", description="Summarised approach one would take to solve the problem")
]

output_parser_reduce = StructuredOutputParser.from_response_schemas(response_schemas_reduce)
reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["question"],  partial_variables={"format_instructions": output_parser_reduce.get_format_instructions()},output_parser=output_parser_reduce)
reduce_chain=LLMChain(llm=llm,prompt=reduce_prompt)

def formatReduce(response):
    formattedStr=""
    for idx,i in enumerate(response):
        questionStr="Question "+ str(idx)
        formattedStr+=questionStr+ "\n"
        formattedStr+="Challenge Summary for " +questionStr + "\n"
        formattedStr+=i["challenge"]
        formattedStr+="\n Approach Summary for " +questionStr + "\n"
        formattedStr+=i["approach"]
        if idx==len(response)-1:
            continue
        formattedStr+='```\n'
    return formattedStr
rank_template = """The following are summaries of examination question, specifically challenge and solution summaries, which are delimited by ```. Considering their challenge and solution summaries, give them a difficulty level from 0 to 10, with 10 being the hardest. Reason why each problem is given a certain difficulty level.
{format_instructions} 
```
{questionSummaries}
```
"""
response_schemas_rank = [
    ResponseSchema(name="question_id", description="Unique identifier for the question"),
    ResponseSchema(name="rank", description="Ranking of the question based on challenge and approach")
]

output_parser_rank = StructuredOutputParser.from_response_schemas(response_schemas_rank)
rank_prompt = PromptTemplate(template=rank_template, input_variables=["questionSummaries"],  partial_variables={"format_instructions": output_parser_rank.get_format_instructions()},output_parser=output_parser_rank)
rank_chain=LLMChain(llm=llm,prompt=rank_prompt)

filename = "MSTQns.txt"
questions = []
with open(filename, 'r') as file:
    for line in file:
        questions.append(line.strip())
inputList=[]
parsedList=[]
for q in questions: #forming required dict format
    entry={}
    entry["question"]=q
    inputList.append(entry)

reduceProblems= reduce_chain.apply_and_parse(inputList)
formatProblem= formatReduce(reduceProblems)
rankings=rank_chain.predict(questionSummaries=formatProblem)
print(rankings)