import ollama
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="llama3.2:1b",
    temperature=0,   
    top_k=20,        
    num_ctx=512,    
    repeat_penalty=1.1,   
    ) 

response = model("Hello, how are you?")
print(response)




















# from typing import Dict, List
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from langchain_community.llms import Ollama
# from langchain.chains import ConversationalRetrievalChain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# import asyncio
# import time
# import hashlib

# # Initialize with performance optimizations
# app = FastAPI(title="Farmer Advisor - Turbo API", version="2.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Turbo Configuration ⚡
# CONFIG = {
#     "model": "llama3.2:1b",
#     "embedding_model": "all-MiniLM-L6-v2",  # Keep your original embeddings
#     "retriever_k": 2,  # Reduced from original
#     "max_history": 2,  # Only keep last 2 exchanges
#     "timeout": 12.0,  # Fail fast
#     "max_tokens": 192,  # Strict output limit
#     "temperature": 0.3  # Slightly higher for better quality
# }

# # Pre-load components
# print("🚀 Turbo Initializing llama3.2:1b...")
# start_time = time.time()

# # 1. Optimized Vector Store
# vector_store = Chroma(
#     collection_name="farm_advicer",
#     embedding_function=HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"]),
#     persist_directory="./chroma_langchain_db",
# )

# # 2. Performance-Tuned LLM
# llm = Ollama(
#     model=CONFIG["model"],
#     temperature=CONFIG["temperature"],
#     num_ctx=768,  # Reduced context window
#     top_k=15,     # Faster sampling
#     repeat_penalty=1.15,
#     num_gpu=1 if torch.cuda.is_available() else 0  # Auto GPU detection
# )

# # 3. Lean Retriever
# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={
#         "k": CONFIG["retriever_k"],
#         "score_threshold": 0.65  # Better quality filtering
#     }
# )

# # 4. Ultra-Compact Prompt
# PROMPT_TEMPLATE = """Answer in MAX 3 bullet points using ONLY these documents:
# {context}

# Question: {question}

# Format:
# - [Key point 1] (Author Year)
# - [Key point 2] 
# - [Climate impact]"""

# rag_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True,
#     max_tokens_limit=CONFIG["max_tokens"],
#     verbose=False,  # Disable internal logging
#     combine_docs_chain_kwargs={
#         "prompt": ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     }
# )

# print(f"✅ Ready in {time.time() - start_time:.2f}s")

# # Turbo Chat Handling
# chat_histories: Dict[str, List] = {}
# connected_clients = set()

# def create_doc_fingerprint(doc):
#     """Super-fast deduplication hash"""
#     title_hash = hash(doc.metadata.get("title", ""))
#     content_hash = hash(doc.page_content[:128])  # First 128 chars only
#     return f"{title_hash}_{content_hash}"

# @app.websocket("/ws")
# async def turbo_websocket(websocket: WebSocket):
#     await websocket.accept()
#     client_id = str(id(websocket))
#     connected_clients.add(websocket)
#     chat_histories[client_id] = []

#     try:
#         while True:
#             # Receive with timeout
#             try:
#                 data = await asyncio.wait_for(
#                     websocket.receive_json(),
#                     timeout=30.0  # Client-side timeout
#                 )
#             except asyncio.TimeoutError:
#                 await websocket.send_json({"error": "Client timeout"})
#                 continue

#             query = data.get("query", "").strip()
#             if not query:
#                 continue

#             start_time = time.perf_counter()
            
#             try:
#                 # Process with aggressive timeout
#                 response = await asyncio.wait_for(
#                     rag_chain.ainvoke({
#                         "question": query,
#                         "chat_history": chat_histories[client_id][-CONFIG["max_history"]:]
#                     }),
#                     timeout=CONFIG["timeout"]
#                 )

#                 # Lightning-fast deduplication
#                 unique_docs = {}
#                 for doc in response["source_documents"]:
#                     if not doc.page_content:
#                         continue
#                     doc_id = create_doc_fingerprint(doc)
#                     if doc_id not in unique_docs:
#                         unique_docs[doc_id] = doc

#                 # Update history
#                 chat_histories[client_id].append((query, response["answer"]))
                
#                 # Turbo response
#                 await websocket.send_json({
#                     "query": query,
#                     "answer": response["answer"],
#                     "retrieved_docs": [
#                         {
#                             "title": doc.metadata.get("title", "N/A"),
#                             "year": doc.metadata.get("yearPublished", ""),
#                         }
#                         for doc in list(unique_docs.values())[:2]  # Top 2 only
#                     ],
#                     "response_time": f"{time.perf_counter() - start_time:.2f}s"
#                 })

#             except asyncio.TimeoutError:
#                 await websocket.send_json({"error": "Processing timeout"})
#             except Exception as e:
#                 await websocket.send_json({"error": f"Processing error: {str(e)}"})

#     except WebSocketDisconnect:
#         pass
#     finally:
#         connected_clients.discard(websocket)
#         chat_histories.pop(client_id, None)

# # Minimal API Endpoints
# @app.get("/")
# def root():
#     return {"status": "Turbo Mode Active"}

# @app.get("/model")
# def model_info():
#     return {
#         "model": CONFIG["model"],
#         "max_history": CONFIG["max_history"],
#         "timeout": CONFIG["timeout"]
#     }