from typing import Dict
from fastapi import FastAPI, WebSocket,WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from utility import fetch_documents, init, chunk_text, load_vector_db,clean_response
import asyncio
import time

# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
import heapq
from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from llama_cpp import Llama
from huggingface_hub import hf_hub_download 
from pathlib import Path

app = FastAPI(
    title="Farmer Advisor - API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

 
#  Retriever
# retriever = vector_store.as_retriever(
#     search_type="mmr", # mmr is computationally expensive
#     search_kwargs={"k": 5, "fetch_k": 5, "lambda_mult": 0.6}
# ) 

MODEL_CACHE = str(Path.home() / ".cache" / "models")  # Better alternative
Path(MODEL_CACHE).mkdir(parents=True, exist_ok=True)
vector_store = load_vector_db()
# model = OllamaLLM(model="llama3.2:1b",
#     temperature=0,   # high is creative but low makes it coherent
#     top_k=20,        
#     num_ctx=512,    
#     repeat_penalty=1.1,   
#     ) 
model = Llama(
    model_path=hf_hub_download(
        repo_id="JayROgada/tendo.gguf",
        filename="tendo.gguf",
        cache_dir=MODEL_CACHE  # Persistent storage on Railway
    ),
    n_ctx=512,            # Matches your num_ctx
    n_threads=4,          # Optimal for Railway's 2vCPU
    temperature=0.0,     
    top_k=20,             
    repeat_penalty=1.1,   #  repetition control
    n_gpu_layers=-1       # Auto-detect GPU layers if available
)


retriever = vector_store.as_retriever(
    search_type="similarity",  
    search_kwargs={"k": 3,           
        }
) 


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


def clean_response(text):
    """Removes Markdown formatting from the model's response."""
    text = re.sub(r"###\s*", "", text)  # Remove headers
    text = re.sub(r"\*\*|\*", "", text)  # Remove bold/italic
    text = text.strip()  # Remove trailing whitespace
    return text

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
 
    combined_corpus_df['is_duplicate'] = combined_corpus_df.apply(
        lambda row: is_similar(row['title'], combined_corpus_df['title'].shift().loc[row.name]), 
        axis=1
    )
    combined_corpus_df = combined_corpus_df[~combined_corpus_df['is_duplicate']]
    combined_corpus_df = combined_corpus_df.drop(columns=['is_duplicate'])
    return combined_corpus_df


def chunk_text(combined_corpus_df):
    combined_corpus_df= remove_duplicates(combined_corpus_df)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_chunks = []
    for _, row in combined_corpus_df.iterrows():
        full_text = row.get("fullText")
        if pd.isna(full_text):
            continue
        chunks = text_splitter.split_text(full_text)
        title = row.get("title", "")
        authors = row.get("authors", "")        
        if isinstance(authors, list):
            authors = ", ".join(author.get("name", "") for author in authors if isinstance(author, dict))
        else:
            authors = str(authors)

        metadata = {
            "title": row.get("title", ""),
            "authors": authors, 
            "publishedDate": row.get("publishedDate", ""),
            "yearPublished": row.get("yearPublished", ""),
            "doi": row.get("doi", ""),
            "publisher": row.get("publisher", ""),
            "documentType": row.get("documentType", ""),
            "fieldOfStudy": row.get("fieldOfStudy", ""),
            "journals": row.get("journals", ""),
            "sourceFulltextUrls": row.get("sourceFulltextUrls", []),  
            "links": row.get("links", []),  
 }

        metadata = simple_filter_metadata(metadata)
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
            if all(isinstance(item, dict) and "name" in item for item in value):
                simple_metadata[key] = ", ".join(item["name"] for item in value)
            else:
                simple_metadata[key] = ", ".join(str(item) for item in value)
        else:
            simple_metadata[key] = str(value)
    return simple_metadata

def create_vector_db(chunks, batch_size=100):
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="farm_advicer",
        embedding_function=embedding_function,
        persist_directory="chroma_langchain_db",  
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
import ast

def parse_links(links_str):
    """
    Parses the links string and extracts download, reader, and thumbnail URLs.
    
    Args:
        links_str (str): A string representation of a list of dictionaries.
    
    Returns:
        dict: A dictionary with 'download', 'reader', and 'thumbnail' URLs.
    """
    parsed_links = []
    
    if links_str and links_str != "n/a":
        try:
            # Convert string to a list of dictionaries
            parsed_links = [ast.literal_eval(item.strip() + "}") if not item.endswith("}") else ast.literal_eval(item.strip()) 
                            for item in links_str.split("},") if item]
        except Exception as e:
            print("Error parsing links:", e)
            return {"download": "N/A", "reader": "N/A", "thumbnail": "N/A"}

    # Extract specific links
    return {
        "download": next((item['url'] for item in parsed_links if item['type'] == 'download'), "N/A"),
        "reader": next((item['url'] for item in parsed_links if item['type'] == 'reader'), "N/A"),
        "thumbnail": next((item['url'] for item in parsed_links if 'thumbnail' in item['type']), "N/A")
    }
 
custom_prompt = ChatPromptTemplate.from_template("""
You are an expert in livestock farming and emissions reduction policies. Your task is to educate farmers on these topics in a clear and practical manner.

You will be given a set of queries related to livestock farming and emissions reduction policies:

<context>
{context}
</context>

Follow these steps:
1️ **Understand the Queries** – Identify key topics, such as livestock emissions, manure management, mitigation strategies, the Paris Agreement, precision livestock farming, and carbon emissions in farming.  
2️ **Explain Why It Matters** – For each query, provide a brief, **farmer-friendly** explanation of why it’s important and how it impacts their work.  
3️ **Give Practical Advice** – Provide **simple, actionable tips** to help farmers adopt sustainable practices and comply with policies.  
4️ **Use a Clear Structure** – Format your response with **headings for each topic** and make it **concise yet informative**.  
5️ **Wrap in Answer Tags** – Place your complete response inside **<answer>** tags.

Example Response Format:
 
** Manure Management & Methane Reduction**  
*Why It Matters:* Managing manure effectively reduces methane, which contributes to climate change.  
*Practical Tip:* Cover manure storage areas to reduce methane emissions by up to 50%.  
*Regulatory Insight:* The Paris Agreement encourages emission reduction in agriculture.  

**🔍 Precision Livestock Farming**  
*Why It Matters:* Using sensors and AI can optimize feed, reducing emissions.  
*Practical Tip:* Invest in precision feeding tools to cut feed waste and emissions.  

""")



# • Use Swahili terms in brackets (e.g., methane = gesi metani)
#
# Initialize RAG Chain once with return_source_documents enabled


rag_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever, 
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

chat_histories: Dict[str, list] = {}
MAX_HISTORY = 5


@lru_cache(maxsize=100)
def cached_rag_invoke(query, chat_history):
    return rag_chain.invoke({
        "question": query,
        "chat_history": chat_history
    })

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.get("/fetch-research")
def get_data(): 
    data = init()
    return {"documents": data}

connected_clients = set()

@app.get('/db')
def get_vector_details():
    return {
        # 'vectorDB':vector_store.index,
        'count':vector_store._collection.count(),
        "Collection Name:": vector_store._collection.name,
        "Collection Metadata:":vector_store._collection.metadata
    }
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    session_id = str(id(websocket))
    chat_histories[session_id] = []
    print(f"Client connected. Active clients: {len(connected_clients)}")
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "").strip()
            
            if not query:
                await websocket.send_json({"error": "Query cannot be empty"})
                continue
            print(f"User Query: {query}")
            start_time = time.perf_counter()
            try:
                response = await rag_chain.ainvoke({
                    "question": query,
                    "chat_history": chat_histories[session_id]
                })                
                retrieved_docs = response.get("source_documents", []) or []                
                unique_docs = {}
                for doc in retrieved_docs:
                    if not doc.page_content or not doc.metadata:
                        continue                        
                    doc_key = (
                        doc.metadata.get('title', '').lower().strip(),
                        doc.metadata.get('doi', '').lower().strip() or hash(doc.page_content[:200])
                    )                    
                    if doc_key not in unique_docs:
                        unique_docs[doc_key] = doc
                context_parts = []
                cited_sources = set()                
                for doc in unique_docs.values():
                    authors = doc.metadata.get('authors', 'Unknown')
                    year = doc.metadata.get('yearPublished', '')
                    title = doc.metadata.get('title', 'Untitled')
                    source_key = f"{authors}-{title}"
                    
                    if source_key not in cited_sources:
                        source_info = f"{authors} ({year}) {title}" if year else f"{authors} {title}"
                        context_parts.append(f"Source: {source_info}\nContent: {doc.page_content[:800]}")
                        cited_sources.add(source_key)

                context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found"                
                # Prepare response
                response_time = time.perf_counter() - start_time
                chat_histories[session_id].append((query, response["answer"]))
                chat_histories[session_id] = chat_histories[session_id][-MAX_HISTORY:]
                # Prepare document snippets for response (top 2 unique docs)
                doc_snippets = []
                for doc in list(unique_docs.values())[:2]:
                    doc_snippets.append({
                        "title": doc.metadata.get("title", "N/A"),
                        "authors": doc.metadata.get("authors", "Unknown"),
                        "publishedDate": doc.metadata.get("publishedDate", "Unknown"),
                        "yearPublished": doc.metadata.get("yearPublished", "Unknown"),
                        "doi": f"https://doi.org/{doc.metadata['doi']}" if doc.metadata.get("doi") not in [None, "N/A"] else "N/A",
                        "publisher": doc.metadata.get("publisher", "Unknown"),
                        "fieldOfStudy": doc.metadata.get("fieldOfStudy", "Unknown"),
                        "links": parse_links(doc.metadata.get("links", "n/a"))  if doc.metadata.get("links", "n/a") and doc.metadata.get("links", "n/a") != "n/a" else "n/a",
                    })
                await websocket.send_json({
                    "query": query,
                    "answer": response["answer"],
                    "retrieved_docs": doc_snippets,
                    "response_time": f"{response_time:.2f} sec",
                    "chat_history": chat_histories[session_id]})
                print(f"answer: {response['answer']}")
                print(f"response_time: {response_time:.2f} sec")
            except Exception as e:
                print(f"Internal Processing Error: {e}")
                await websocket.send_json({"error": "Internal error occurred."})
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        chat_histories.pop(session_id, None)
        print(f"Client removed. Active clients: {len(connected_clients)}")