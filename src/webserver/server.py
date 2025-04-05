from fastapi import FastAPI
from webserver.constants import *
from pydantic import BaseModel
from provider.llm_provider import LLMOllama

class RequestPayload(BaseModel):
    model: str
    prompt: str
    images: list[str]
    stream: bool = False

app = FastAPI(
    title="LLM inference API",
    description="API for LLM inference",
    version="0.1",
)

@app.get("/")
def read_root():
    return {"version": WEB_SERVER_VERSION}

@app.post("/api/generate")
def generate_response(request: RequestPayload):
    """Generate a response from the LLM model."""
    provider = LLMOllama("http://127.0.0.1:11434", model=request.model, images=request.images)
    response = provider.generate_response(request.prompt)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)