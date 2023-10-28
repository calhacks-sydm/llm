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
if __name__ == "__main__":
    response=run_llm("What is Dijkstra's algorithm")
    print(response["answer"]) #one answer, multiple source_documents (name is shortform)