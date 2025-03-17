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
from concurrent.futures import ProcessPoolExecutor
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json
 
from fuzzywuzzy import fuzz
from tqdm import tqdm
# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")


vector_store =''

def query_api(query:str, scrollId=None, ):
    headers={"Authorization":"Bearer "+API_KEY}
    if not scrollId:
        response = requests.get(f"{BASE_URL}?q={query}&limit={DOCUMENT_FETCH_SIZE}&scroll=true",headers=headers)
    else:
        response = requests.get(f"{BASE_URL}?q={query}&limit={DOCUMENT_FETCH_SIZE}&scrollId={scrollId}",headers=headers)
    return response.json(), response.elapsed.total_seconds()
 
def to_data_frame(data:List[dict]):
    return pd.DataFrame(data)

def fetch_documents(): 
    '''
    Fetch data using dictionary comprehension
    Convert API results to DataFrames
    Combine DataFrames efficiently
    '''
    with ThreadPoolExecutor() as executor:  # Using threading instead of multiprocessing
        query_results = list(executor.map(query_api, queries.values()))  
    query_results = [result[0] for result in query_results]  

    dataframes = [to_data_frame(result) for result in query_results]
    combined_corpus_df = pd.concat(dataframes, ignore_index=True)   
    return combined_corpus_df

# remove duplicates from data
def remove_duplicates(combined_corpus_df):
    # Normalize data
    combined_corpus_df['title'] = combined_corpus_df['title'].str.lower().str.strip()
    combined_corpus_df['abstract'] = combined_corpus_df['abstract'].str.lower().str.strip()

    # Extract author names from dictionaries and join them
    combined_corpus_df['authors'] = combined_corpus_df['authors'].apply(
        lambda x: ", ".join(sorted([author["name"] for author in x])) if isinstance(x, list) else x
    )

    combined_corpus_df['fullText'] = combined_corpus_df['fullText'].str.lower().str.strip()

    # Remove exact duplicates
    combined_corpus_df = combined_corpus_df.drop_duplicates(subset=['title', 'abstract', 'authors', 'fullText'])

    # Function to check similarity
    def is_similar(text1, text2, threshold=90):
        if pd.isna(text1) or pd.isna(text2):  # Handle NaN values
            return False
        return fuzz.ratio(text1, text2) >= threshold

    # Compare each title with the title of the previous row
    combined_corpus_df['is_duplicate'] = combined_corpus_df.apply(
        lambda row: is_similar(row['title'], combined_corpus_df['title'].shift().loc[row.name]), 
        axis=1
    )

    # Remove near-duplicates
    combined_corpus_df = combined_corpus_df[~combined_corpus_df['is_duplicate']]

    # Drop the temporary 'is_duplicate' column
    combined_corpus_df = combined_corpus_df.drop(columns=['is_duplicate'])

    # Check the shape of the deduplicated DataFrame
    return combined_corpus_df


def chunk_text(combined_corpus_df):
    combined_corpus_df= remove_duplicates(combined_corpus_df)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_chunks = []
    
    # Iterate over each document (row) in the dataframe
    for _, row in combined_corpus_df.iterrows():
        full_text = row.get("fullText")
        if pd.isna(full_text):
            continue
        # Split the full text into chunks
        chunks = text_splitter.split_text(full_text)
        
        # Extract and process metadata fields
        title = row.get("title", "")
        authors = row.get("authors", "")
        # Convert authors (a list of dicts) into a comma-separated string if needed.
    # Extract authors as a comma-separated string
        if isinstance(authors, list):
            authors = ", ".join(author.get("name", "") for author in authors if isinstance(author, dict))
        else:
            authors = str(authors)

        # Extract metadata fields with default empty strings
        metadata = {
            "title": row.get("title", ""),
            "authors": authors,
            # "abstract": row.get("abstract", ""),
            "publishedDate": row.get("publishedDate", ""),
            "yearPublished": row.get("yearPublished", ""),
            "doi": row.get("doi", ""),
            "publisher": row.get("publisher", ""),
            "documentType": row.get("documentType", ""),
            "fieldOfStudy": row.get("fieldOfStudy", ""),
            "journals": row.get("journals", ""),
            "sourceFulltextUrls": row.get("sourceFulltextUrls", []),  # List of URLs
            "links": row.get("links", []),  # Additional links, if available
 }

        # Filter metadata to ensure all values are simple types (str, int, etc.)
        metadata = simple_filter_metadata(metadata)
        
        # Attach the metadata to each text chunk
        for chunk in chunks:
            all_chunks.append({"text": chunk, "metadata": metadata})
            
    print(f"Total number of chunks: {len(all_chunks)}")
    return all_chunks

def simple_filter_metadata(metadata: dict) -> dict:
    """
    Convert complex metadata values into simple types (str, int, float, or bool).
    For example, if authors is a list of dicts, join the names into a string.
    """
    simple_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            simple_metadata[key] = value
        elif isinstance(value, list):
            # Check if it's a list of dicts with a 'name' key.
            if all(isinstance(item, dict) and "name" in item for item in value):
                simple_metadata[key] = ", ".join(item["name"] for item in value)
            else:
                # Otherwise, just join as strings.
                simple_metadata[key] = ", ".join(str(item) for item in value)
        else:
            simple_metadata[key] = str(value)
    return simple_metadata

def create_vector_db(chunks, batch_size=100):
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="farm_advicer",
        embedding_function=embedding_function,
        persist_directory="chroma_langchain_db",  # Use a consistent directory, e.g., VECTOR_DB_DIR = "./chroma_langchain_db"
    )
    # Process chunks in batches
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        metadatas = [chunk["metadata"] for chunk in batch]
        vector_store.add_texts(texts, metadatas=metadatas)
    print("Vector DB successfully created and persisted.")
    return vector_store


def load_vector_db():
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="farm_advicer",
        embedding_function=embedding_function,
        persist_directory="./chroma_langchain_db",  # Load from disk
    )
    return vector_store 


def init(): 
    '''
    - fetch documents
    - save locally
    - chunking
    - embed to vector_db (You can append more data to it) 
    '''
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
    print(f"✅ Text chunked in {chunk_end - chunk_start:.2f} seconds")
    print("Creating vector database...")
    vector_start = time.time()
    vector_store = create_vector_db(chunks)  
    chroma_vector_DB_status = "created"
    vector_end = time.time()
    print(f"✅ Vector database created in {vector_end - vector_start:.2f} seconds")
    total_time = time.time() - start_time 
    print("*************************************************")
    print(f"🚀 Process completed in {total_time:.2f} seconds")
    print("*************************************************")
    data = {
        "time_taken": total_time,   
        "chunk":len(chunks),
        "chroma_vector_DB_status":chroma_vector_DB_status,
        "data": json.loads(df.to_json(orient="records")) 
    }
    return data

