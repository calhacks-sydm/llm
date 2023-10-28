import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)
INDEX_NAME = "calhacks-cs170"

def ingest_docs():
    loader = PyPDFLoader("cs170TB.pdf")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(raw_documents)
    # for doc in documents:
    #     new_url = doc.metadata["source"]
    #     doc.metadata.update({"source": new_url})
    # embeddings = OpenAIEmbeddings()
    # print(f"Going to add {len(documents)} to Pinecone")
    # Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    # print("****Loading to vectorestore done ***")

if __name__ == "__main__":
    ingest_docs()