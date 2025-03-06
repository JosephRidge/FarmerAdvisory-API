import pandas as pd
import requests
import os  
from constants import queries, DATA_DIR,DOCUMENT_FETCH_SIZE
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from concurrent.futures import ProcessPoolExecutor
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
 
# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")


def query_api(query:str, scrollId=None, ):
    headers={"Authorization":"Bearer "+API_KEY}
    if not scrollId:
        response = requests.get(f"{BASE_URL}?q={query}&limit={DOCUMENT_FETCH_SIZE}&scroll=true",headers=headers)
    else:
        response = requests.get(f"{BASE_URL}?q={query}&limit={DOCUMENT_FETCH_SIZE}&scrollId={scrollId}",headers=headers)
    return response.json(), response.elapsed.total_seconds()
 
def to_data_frame(data:List[dict]):
    return pd.DataFrame(data) #[0]['results']

def fetch_documents(): 
    '''
    Fetch data using dictionary comprehension
    Convert API results to DataFrames
    Combine DataFrames efficiently
    '''
    with ThreadPoolExecutor() as executor:  # Use threading instead of multiprocessing
        query_results = list(executor.map(query_api, queries.values()))  

    # Ensure only JSON data is passed to to_data_frame()
    query_results = [result[0] for result in query_results]  

    dataframes = [to_data_frame(result) for result in query_results]
    combined_corpus_df = pd.concat(dataframes, ignore_index=True)   
    return combined_corpus_df

def chunk_text(combined_corpus_df):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    print(combined_corpus_df)
    with ThreadPoolExecutor() as executor:
        all_chunks = list(executor.map(text_splitter.split_text, combined_corpus_df['fullText'].dropna()))
    # Flatten the list
    all_chunks = [chunk for sublist in all_chunks for chunk in sublist]
    print(f"Total number of chunks: {len(all_chunks)}")
    return all_chunks


def create_vector_db(chunks, batch_size=100):  # Process in small batches
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="farm_advisor",
        embedding_function=embedding_function,
        persist_directory="./farmer_chroma_langchain_db",
    )
    # Insert chunks in batches
    for i in tqdm(range(0, len(chunks), batch_size)):
        vector_store.add_texts(chunks[i:i + batch_size])
    print("Vector DB successfully created and persisted.")

def load_vector_db():
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="farm_advisor",
        embedding_function=embedding_function,
        persist_directory="./chroma_langchain_db",  # Load from disk
    )
    return vector_store 

vector_store =''


import json


def init(): 
    '''
    - fetch documents
    - save locally
    - chunking
    - embed to vector_db (You can append more data to it) 
    '''
    # df = fetch_documents()
    # file_path = os.path.join(DATA_DIR, "data.csv")
    # df.to_csv(file_path, index=False)
    # chunks = chunk_text(df)    
    # create_vector_db(chunks)
    start_time = time.time()
    
    print("*************************************************")
    print("Fetching documents...")
    fetch_start = time.time()
    df = fetch_documents() 
    df = df['results'].apply(pd.Series)
    fetch_end = time.time() 
    print(f"✅ Documents fetched in {fetch_end - fetch_start:.2f} seconds")
    
    
    print("Chunking text...")
    chunk_start = time.time()
    chunks = chunk_text(df)    
    chunk_end = time.time()
    # print(f"✅ Text chunked in {chunk_end - chunk_start:.2f} seconds")
    # print("Creating vector database...")
    # vector_start = time.time()
    # vector_store = create_vector_db(chunks) 
    # vector_end = time.time()
    # print(f"✅ Vector database created in {vector_end - vector_start:.2f} seconds")
    
    total_time = time.time() - start_time 
    print("*************************************************")
    print(f"🚀 Process completed in {total_time:.2f} seconds")
    print("*************************************************")
    data = {
        "time_taken_to_get_docs": total_time,   
        "chunk":len(chunks),
        "data": json.loads(df.to_json(orient="records")) 
    }
    return data