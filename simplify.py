from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

import os
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key= os.getenv('OPENAI_API_KEY')
#COMMENTS

#if too many subproblems the value will be empty

# question="Wilson’s Werewolves (15 points) nfriends are playing a game. kof them are werewolves, while the other n−kof them are villagers. The game is set in a cave with some number of rooms, and rooms may be connected by corridors. At the start of the game, there is a set of broken rooms in the cave that require repairs, and each of the players spawns in a random room. There is no limit on the number of players that can spawn ineach room, and it is possible for a player to spawn in a broken room. Everyone then rushes to theclosest broken room (determined by number of corridors to get to the room) to either help fix the cave (villager), or pretend to help fix the cave (werewolf). We want to determine how many villagers are in danger because they end up in the same broken room as a werewolf. Assume that the cave can be represented as an unweighted, undirected graph G= (V, E), where the vertices represent rooms and the edges represent corridors connecting rooms. The set of broken rooms is B⊆V. Further assume that all the players take the shortest path to the closest broken room, and there are no ties. Finally, assume that every room has the capacity to hold any number of players. Design an efficient algorithm to solve this problem and provide its runtime in terms of any of n, k,|V|,|E|,|B|;proof of correctness is not required."
# answer="Algorithm: Create an auxiliary node sand create edges between sand each broken room b∈B. Then, run BFS from source node sto generate the shortest paths between each player and the closest broken room. From here, we already know which player ends up in which broken room since the BFS will generate a tree. We can traverse the tree using DFS, keeping track of the current broken room, and any villagers or werewolves we encounter will end up in that broken room. Thus, we go througheach broken room, and if there is a werewolf that ends up there, we add the number of villagers to our answer. Runtime: The initial BFS is O(|V|+|E|), and the final summation is O(|B|) =O(|V|). Thus, the overall runtime is O(|V|+|E|)"

# response_schemas = [
#     ResponseSchema(name="hint", description="hint to solving the problem"),
#     ResponseSchema(name="subproblem_1", description="sub-problems from breaking down the problems"),
#     ResponseSchema(name="subproblem_2", description="sub-problems from breaking down the problems"),
#     ResponseSchema(name="subproblem_3", description="sub-problems from breaking down the problems"),
#     ResponseSchema(name="subproblem_4", description="sub-problems from breaking down the problems"),
#     ResponseSchema(name="subproblem_5", description="sub-problems from breaking down the problems"),
#     ResponseSchema(name="subproblem_6", description="sub-problems from breaking down the problems"),
#     ResponseSchema(name="subproblem_7", description="sub-problems from breaking down the problems"),
#     ResponseSchema(name="subproblem_8", description="sub-problems from breaking down the problems"),
#     ResponseSchema(name="subproblem_9", description="sub-problems from breaking down the problems"),
#     ResponseSchema(name="subproblem_10", description="sub-problems from breaking down the problems")
# ]

response_schemas = [
    ResponseSchema(name="hint", description="Hints or clues related to the main problem"),
    ResponseSchema(name="sub_problem 1", description="Contains 'title' for brief overview and 'description' for detailed info of the first sub-problem."),
    ResponseSchema(name="sub_problem 2", description="Contains 'title' for brief overview and 'description' for detailed info of the second sub-problem.")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="You are a student learning assistant. For a given question and its answer, generate a hint for student who may be stuck solving this problem and break this subproblem down into simpler problems to increase the ease of solvin .\n{format_instructions}\n Given question: {question}\n Given answer: {answer} ",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)
model = OpenAI(temperature=0, verbose=True)
def runSimplifyllm(question, answer):
    _input = prompt.format_prompt(question=question, answer=answer)
    output = model(_input.to_string())
    return output_parser.parse(output)