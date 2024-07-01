from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "langchain-doc-index"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    # loader = ReadTheDocsLoader(
    #     "D:\vector_emb\assets"
    # )
    # raw_documents = loader.load()
        
    loader = PyPDFLoader("assets/resume.pdf")
    pages = loader.load_and_split()
    print(f"loaded {len(pages)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(pages)
    print(f"Going to add {len(documents)} to Pinecone")
    for document in documents:
        new_content = document.page_content.encode("UTF-8")
        unicode_string = new_content.decode('utf-8')
        document.page_content = unicode_string
    PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorstore done ***")

if __name__ == "__main__":
    ingest_docs()