from fastapi import FastAPI
from src.webserver.constants import *
from pydantic import BaseModel
from src.pipeline.pipeline import Pipeline
from src.configuration.configdb import ConfigDB
import logging
import os
from src.configuration.constants import CONFIG_DB_PATH, CONFIG_DB_NAME

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

db = ConfigDB()
# Get all files in the directory
config_files = [f for f in os.listdir(CONFIG_DB_PATH) if os.path.isfile(os.path.join(CONFIG_DB_PATH, f)) and f.endswith(".json")]
logger.info(f"Config files found: {config_files}")
assert len(config_files) > 0, "No config files found in the configuration directory"

db.setup(CONFIG_DB_PATH, CONFIG_DB_NAME, config_files)

# Initialize the pipeline
pipeline = Pipeline()
try:
    pipeline.setup()
except Exception as e:
    logger.error(f"Error initializing pipeline: {e}")

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
