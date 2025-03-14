from typing import Dict
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from utility import fetch_documents, init, chunk_text, load_vector_db
import asyncio
import time

from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

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
model = Ollama(model="llama3.2")

# Define Retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.6}
)

# Initialize RAG Chain once with return_source_documents enabled
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    return_source_documents=True
)

# Chat history store (in-memory)
chat_histories: Dict[str, list] = {}
MAX_HISTORY = 10

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.get("/fetch-research")
def get_data(): 
    data = init()
    return {"documents": data}

connected_clients = set()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    session_id = str(id(websocket))  # Unique session per client
    chat_histories[session_id] = []  # Initialize chat history

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

            # Invoke RAG chain with history; now it will return source documents as well.
            response = await asyncio.to_thread(
                rag_chain.invoke, {
                    "question": query,
                    "chat_history": chat_histories[session_id]
                }
            )

            # Track response time
            response_time = time.perf_counter() - start_time
            print(f"Response Time: {response_time:.2f} sec")

            # Update chat history (keep only the latest N messages)
            chat_histories[session_id].append((query, response["answer"]))
            chat_histories[session_id] = chat_histories[session_id][-MAX_HISTORY:]

            # Process retrieved docs if available; adjust the metadata key if needed.
            retrieved_docs = response.get("source_documents", [])
            # Here we extract a snippet and use the title as a source if available.
            doc_snippets = [
                {"text": doc.page_content[:200], "source": doc.metadata.get("title", "N/A")}
                for doc in retrieved_docs
            ]

            # Send JSON response with the answer and retrieved docs metadata.
            await websocket.send_json({
                "answer": response["answer"],
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
