from typing import Union

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from utility import fetch_documents, init
from contextlib import asynccontextmanager
import asyncio


# lifespan to run task at start-up
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("API is initializing... 🚀")
#     await init()
#     yield  # API becomes available immediately
#     print("API shutdown...")

app = FastAPI(
    title="Farmer Advisor - API",
    version="1.0" )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/fetch-data")
def get_data():
    return { "documents":fetch_documents() }

@app.get("/fetch-research")
def get_data():
    return { "documents":init() }


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "query": q}


# Storage for task results
# task_results = {"data": None}

# async def fetch_data():
#     await asyncio.sleep(8.5)  # Simulating a long operation
#     task_results["data"] = {"message": "Data fetched"}  # ✅ Store the result

# @app.get("/data")
# async def get_data(background_tasks: BackgroundTasks):
#     task_results["data"] = None  # Reset before processing
#     background_tasks.add_task(fetch_data)  # ✅ Schedule task
#     return {"status": "Processing started, check later"}

# @app.get("/data/result")
# async def get_data_result():
#     if task_results["data"]:# ✅ Return the fetched data
#         return task_results["data"]  
#     return {"status": "Still processing..."}