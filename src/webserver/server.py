from fastapi import FastAPI
from src.webserver.constants import *
from pydantic import BaseModel
from src.pipeline.pipeline import Pipeline
from src.configuration.configdb import ConfigDB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequestPayload(BaseModel):
    model: str
    prompt: str
    images: list[str]
    session_id: str

# Initialize the FastAPI app
app = FastAPI(
    title="LLM inference API",
    description="API for LLM inference",
    version="0.1",
)

# Setup config DB. TODO: refactor, read from file and configure
CONFIG_DB_PATH = "./deploy/configuration"
CONFIG_DB_NAME = "config.db"
CONFIG_FILES = ["dataset_config.json", "pipeline_config.json", "ollama_config.json"]

db = ConfigDB()
db.setup(CONFIG_DB_PATH, CONFIG_DB_NAME, CONFIG_FILES)

# Initialize the pipeline
pipeline = Pipeline()
pipeline.setup()

@app.get("/")
def read_root():
    return {"version": WEB_SERVER_VERSION}

@app.post("/api/generate")
def generate_response(request: RequestPayload):
    """Generate a response from the LLM model."""
    logging.info(f"Received request: {request}")    
    response = pipeline.generate_response(request.prompt, request.images, request.session_id)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.webserver.server:app", host="0.0.0.0", port=8000, reload=False)
