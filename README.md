# llm-context-sdk

**llm-context-sdk** is a Python SDK for managing and augmenting prompt context for large language models (LLMs). It provides utilities to configure LLM providers, load and query document corpora, and build pipelines for context-aware LLM inference.

## Features

- Pluggable LLM provider support (OpenAI, LLaMa, Ollama, Nvidia, etc.)
- Document loading and vector store integration for retrieval-augmented generation (RAG)
- Session management to maintain chat histories
- FastAPI-based inference API server

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Configuration files for LLM providers, pipeline settings, and datasets are located in the `deploy/configuration` directory. Use these JSON files along with `config.db` to define:

- LLM provider settings (`openai_config.json`, `llama_cpp_config.json`, etc.)
- Pipeline parameters (`pipeline_config.json`)
- Dataset retrieval configurations (`dataset_config.json`)

## Usage

### Running the API Server

```bash
./run.sh
```

This starts a FastAPI server on `http://0.0.0.0:8000` with endpoints:

- `GET /` - Health check and version
- `POST /api/generate` - Generate a response given a prompt payload

### Example Client

```python
import requests

payload = {
    "model": "openai",
    "prompt": "Explain the benefits of context-aware LLM applications.",
    "images": [],
    "session_id": "session-1"
}

response = requests.post("http://localhost:8000/api/generate", json=payload)
print(response.json())
```

## Contributing

Please open issues and pull requests for bug fixes or improvements.
