from langchain import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key= os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

explain_template = """You are a helpful teaching assistant. This is the problem {problem} and its solution {solution}. You will explain the solution in context of the problem in a concise and easy to understand manner. If it helps in understanding, bring in analogies."""

explain_prompt = PromptTemplate(template=explain_template, input_variables=["problem","solution"])
explain_chain=LLMChain(llm=llm,prompt=explain_prompt)
# problem="Wilson’s Werewolves (15 points) n friends are playing a game. k of them are werewolves, while the other n − k of them are villagers. The game is set in a cave with some number of rooms, and rooms may be connected by corridors. At the start of the game, there is a set of broken rooms in the cave that require repairs, and each of the players spawns in a random room. There is no limit on the number of players that can spawn in each room, and it is possible for a player to spawn in a broken room. Everyone then rushes to the closest broken room (determined by number of corridors to get to the room) to either help fix the cave (villager), or pretend to help fix the cave (werewolf). We want to determine how many villagers are in danger because they end up in the same broken room as a werewolf. Assume that the cave can be represented as an unweighted, undirected graph G = (V,E), where the vertices represent rooms and the edges represent corridors connecting rooms. The set of broken rooms is B ⊆ V . Further assume that all the players take the shortest path to the closest broken room, and there are no ties. Finally, assume that every room has the capacity to hold any number of players. Design an efficient algorithm to solve this problem and provide its runtime in terms of any of n, k, |V |, |E|, |B|; proof of correctness is not required."
# solution="Create an auxiliary node s and create edges between s and each broken room b ∈ B. Then, run BFS from source node s to generate the shortest paths between each player and the closest broken room. From here, we already know which player ends up in which broken room since the BFS will generate a tree. We can traverse the tree using DFS, keeping track of the current broken room, and any villagers or werewolves we encounter will end up in that broken room. Thus, we go through each broken room, and if there is a werewolf that ends up there, we add the number of villagers to our answer. Runtime: The initial BFS is O(|V | + |E|), and the final summation is O(|B|) = O(|V |). Thus, the overall runtime is O(|V | + |E|)"
def runExplainerllm(problem,solution):
    explanation=explain_chain.predict(problem=problem, solution=solution)
    return(explanation)