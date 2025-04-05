from abc import ABC, abstractmethod
from loguru import logger
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
import ollama

DEFAULT_TEMPERATURE = 0.6

class LLMProvider(ABC):
    """The LLM model."""
    def __init__(self, base_url="http://localhost:11434"):
        """Initialize the default LLM model."""
        self.llm = ChatOpenAI(base_url=base_url, temperature=0.6, api_key=None)

    def generate_response(self, model, prompt, images=None):
        text = ""
        """Generate a response from the LLM model."""
        for chunk in self.llm.stream(prompt):
            text += str(chunk.content)
        return text

    def stream_response(self, model, prompt, images=None):
        """Stream a response from the LLM model."""
        for chunk in self.llm.stream(prompt):
            yield chunk.content

    @abstractmethod
    def get_models_list(self):
        """Get the list of models."""
        pass
    

class LLMOllama(LLMProvider):
    """The Ollama LLM model."""
    def __init__(self, base_url="http://localhost:11434", model=None, temperature=DEFAULT_TEMPERATURE):
        """Initialize the LLM model."""
        self.base_url = base_url
        models_list = self.get_models_list()
        if model is None:
            logger.info(f"Setting up default model {models_list[0]}")
            self.llm = ChatOllama(model=models_list[0], base_url=base_url, temperature=temperature)   
        else:
            logger.info(f"Setting up model from session state {model}")
            self.llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)

    def set_model(self, model):
        """Set the model."""
        self.llm.model = model
        logger.info(f"Setting model to {model}")
        self.llm = ChatOllama(model=self.llm.model, base_url=self.llm.base_url, temperature=DEFAULT_TEMPERATURE)

    def get_models_list(self):
        """Get the list of models."""
        model_list = []
        client = ollama.Client(host=self.base_url)
        response = client.list()
        # logger.info(f"Response: {response}")

        models = response["models"]
        for model in models:
            model_list.append(model["name"])

        return model_list
    
class LLMLlamaCpp(LLMProvider):
    """The Llama.cpp LLM model."""
    def __init__(self, base_url="http://localhost:8080"):
        """Initialize the LLM model."""
        self.llm = ChatOpenAI(base_url=base_url, temperature=0.6, api_key=None)