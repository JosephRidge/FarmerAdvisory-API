from typing import Dict
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from utility import fetch_documents, init, chunk_text, load_vector_db
import asyncio
import time

from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
import heapq
from functools import lru_cache

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


# Load vector DB & model once
vector_store = load_vector_db()
model = Ollama(model="llama3.2:1b") 

# Define Retriever
retriever = vector_store.as_retriever(
    search_type="mmr", # mmr is computationally expensive
    search_kwargs={"k": 5, "fetch_k": 5, "lambda_mult": 0.6}
) 

# retriever = vector_store.as_retriever(
#     search_type="similarity",  # Faster than mmr
#     search_kwargs={"k": 3}
# )
custom_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant specializing in agriculture and climate change.  
Your goal is to provide **clear, concise, and well-structured answers** strictly based on the retrieved documents.  

### 🔹 Rules for Answering:  
1. **Do not fabricate or assume citations**—only use references from the retrieved documents.  
2. **Write a structured, citation-free response** in the main answer.  
3. **Do not insert inline citations.** Instead:  
   - Format the answer naturally, without breaking readability.  
   - List all referenced sources separately under a "References" section.  
4. **If the retrieved documents do not contain enough information**, explicitly state:  
   ➝ *"The provided context does not contain enough information to answer this question."*  
5. **Strictly follow APA citation rules in the "References" section.**  
6. **Your response will be rejected if it violates these rules.**  

---

### 📌 Retrieved Documents (For Reference):  
{context}  

### ❓ Question:  
{question}  

### ✅ Answer Format:  
#### 🌍 Response:  
[Your detailed, citation-free response goes here.]  

#### 📚 References:  
List only the retrieved documents with metadata. Format:  
- Author(s). (Year). *Title of the document*. Source/Journal.  
""")






# Initialize RAG Chain once with return_source_documents enabled
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,#retriever,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

# Chat history store (in-memory)
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
        "📌 Collection Name:": vector_store._collection.name,
        "📌 Collection Metadata:":vector_store._collection.metadata
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
            query = data.get("query", "")

            if not query:
                await websocket.send_json({"error": "Query cannot be empty"})
                continue

            print(f"User: {query}")

            start_time = time.perf_counter()

            # Retrieve documents first (if needed for debugging or custom processing)
            retrieved_docs = await retriever.ainvoke(query)

            # Construct the context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Prepare input for the RAG chain
            input_data = {
                "question": query,
                "chat_history": chat_histories[session_id],
                "context": context  # Add the constructed context explicitly
            }

            # Invoke RAG chain with the prepared input
            response = await rag_chain.ainvoke(input_data)

            # Track response time
            response_time = time.perf_counter() - start_time
            print(f"Response Time: {response_time:.2f} sec")

            # Update chat history (keep only the latest N messages)
            chat_histories[session_id].append((query, response["answer"]))
            chat_histories[session_id] = chat_histories[session_id][-MAX_HISTORY:]

            # Process retrieved docs if available; adjust the metadata key if needed.
            retrieved_docs = response.get("source_documents", [])

            # Rank based on title and content relevance (hybrid ranking)
            query_lower = query.lower()

            # Use heapq for efficient sorting
            top_k = heapq.nlargest(
                2, retrieved_docs, 
                key=lambda doc: (
                    (query_lower in doc.metadata.get("title", "").lower()) * 2
                    + (query_lower in doc.page_content.lower())
                )
            )

            # Extract snippets and metadata
            doc_snippets = [
                {      
                    # "text": doc.page_content[:500],
                    "title": doc.metadata.get("title", doc.metadata.get("source", "N/A")),  # Use "source" if title is missing
                    "authors": doc.metadata.get("authors", "Unknown"),
                    "publishedDate": doc.metadata.get("publishedDate", "Unknown"),
                    "yearPublished": doc.metadata.get("yearPublished", "Unknown"),
                    "doi": f"https://doi.org/{doc.metadata['doi']}" if "doi" in doc.metadata and doc.metadata["doi"] != "N/A" else "N/A",
                    "publisher": doc.metadata.get("publisher", "Unknown"),
                    "fieldOfStudy": doc.metadata.get("fieldOfStudy", "Unknown"),
                    "links": doc.metadata.get("links", "n/a"),
                }
                for doc in top_k
            ]

            # Send JSON response with the answer and retrieved docs metadata
            await websocket.send_json({
                "query": query,
                "answer": response["answer"],  # Now response is defined
                "retrieved_docs": doc_snippets,
                "response_time": f"{response_time:.2f} sec",
                "chat_history": chat_histories[session_id]
            })

    except Exception as e:
        print(f"Client disconnected: {e}")

    finally:
        connected_clients.remove(websocket)
        chat_histories.pop(session_id, None)
        print(f"Client removed. Active clients: {len(connected_clients)}")
