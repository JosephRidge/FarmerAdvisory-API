from typing import Union

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from utility import fetch_documents, init, chunk_text
from contextlib import asynccontextmanager
import asyncio

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

@app.get("/fetch-research")
def get_data(): 
    data = init()
    return { "documents":data }
