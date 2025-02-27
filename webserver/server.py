from fastapi import FastAPI
from constants import *

app = FastAPI(
    title="LLM inference API",
    description="API for LLM inference",
    version="0.1",
)
@app.get("/")
def read_root():
    return {"version": WEB_SERVER_VERSION}




