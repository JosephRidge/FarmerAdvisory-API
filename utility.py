import pandas as pd
import requests
import os  
from constants import queries, DATA_DIR
import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA

import aiofiles
# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")


def query_api(query:str, scrollId=None, ):
    headers={"Authorization":"Bearer "+API_KEY}
    if not scrollId:
        response = requests.get(f"{BASE_URL}?q={query}&limit=70&scroll=true",headers=headers)
    else:
        response = requests.get(f"{BASE_URL}?q={query}&limit=70&scrollId={scrollId}",headers=headers)
    return response.json(), response.elapsed.total_seconds()

def to_data_frame(data:List[dict]):
    return pd.DataFrame(data[0]['results'])


def fetch_documents(): 
    '''
    Fetch data using dictionary comprehension
    Convert API results to DataFrames
    Combine DataFrames efficiently
    '''
    query_results = {key: query_api(query) for key, query in queries.items()}
    dataframes = {key: to_data_frame(result) for key, result in query_results.items()}
    combined_corpus_df = pd.concat(dataframes.values(), ignore_index=True)
    return combined_corpus_df

DATA_DIR = "documents"  # Local storage

# async def fetch_text_documents():
#      with aiofiles.open(file_path, "w") as f:
#         await f.write("This is a fetched document for RAG processing.")


def chunk_text(combined_corpus_df:List[dict]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    combined_corpus_df['chunks'] = combined_corpus_df['fullText'].dropna().apply(text_splitter.split_text) 
    all_chunks = [chunk for sublist in combined_corpus_df['chunks'].dropna() for chunk in sublist]
    len(all_chunks)  # Total number of text chunks
    print(f"Total number of chunks: {len(all_chunks)}") # remove when cleaning
    return all_chunks

def create_vector_db(chunks):
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
    collection_name="farm_advicer",
    embedding_function=embedding_function,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, persistence
    )
    vector_store.add_texts(chunks) # took 28min for 231 articles, and Total number of chunks: 27675


def init(): 
    '''
    - fetch documents
    - save locally
    - chunking
    - embed to vector_db (You can append more data to it) 
    '''
    df = fetch_documents()
    file_path = os.path.join(DATA_DIR, "data.csv")
    df.to_csv(file_path, index=False)
    chunks = chunk_text(df)    
    create_vector_db(chunks)
