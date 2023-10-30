from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone

import os
from dotenv import load_dotenv
load_dotenv()
import pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT_REGION"),
)
INDEX_NAME = "calhacks-cs170"

#COMMENTS
# RetrievalQA does not support dictionary output
# ConversationalRetrievalChain requires chat_history or memory whose key is chat_history
# ConversationBufferMemory only accepts string
# tune params for retrieval at https://python.langchain.com/docs/use_cases/question_answering/vector_db_qa

def run_llm(query):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    llm = ChatOpenAI(
        verbose=True,
        temperature=0,
    )
    qa = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="map_reduce", retriever=docsearch.as_retriever(), return_source_documents=True)
    return qa({"question": query, "chat_history":""}) #no memory

def load_chain(model,K):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    llm = ChatOpenAI(
        temperature=0,
        model_name=model
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=docsearch.as_retriever(search_kwargs={"k": K}),
        return_source_documents=True,
        get_chat_history=lambda h : h
    )
    return qa 
if __name__ == "__main__":
    prompt="This is the solution to a question: Create an auxiliary node s and create edges between s and each broken room b ∈ B. Then, run BFS from source node s to generate the shortest paths between each player and the closest broken room. From here, we already know which player ends up in which broken room since the BFS will generate a tree. We can traverse the tree using DFS, keeping track of the current broken room, and any villagers or werewolves we encounter will end up in that broken room. Thus, we go through each broken room, and if there is a werewolf that ends up there, we add the number of villagers to our answer. Runtime: The initial BFS is O(|V | + |E|), and the final summation is O(|B|) = O(|V |). Thus, the overall runtime is O(|V | + |E|). What can i refer to understand this better?"
    response=run_llm(prompt)
    print(response["answer"]) #one answer, multiple source_documents (name is shortform)