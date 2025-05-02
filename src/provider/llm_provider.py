from abc import ABC, abstractmethod
from loguru import logger
from langchain_ollama import ChatOllama
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
from src.provider.constants import LLMProviderType
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import ollama
import os

DEFAULT_TEMPERATURE = 0.6

class LLMProvider(ABC):
    """The LLM model."""
    def __init__(self, base_url="http://localhost:11434"):
        """Initialize the default LLM model."""
        self.llm = ChatOpenAI(base_url=base_url, temperature=0.6, api_key=None)
        self.model_map: dict = {
            LLMProviderType.OPENAI.value: LLMOpenAI,
            LLMProviderType.LLAMA_CPP.value: LLMLlamaCpp,
            LLMProviderType.OLLAMA.value: LLMOllama,
            LLMProviderType.NVIDIA.value: LLMNvidia
        }

    def generate_response(self, prompt, images=None):
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

    @staticmethod
    def instantiate(provider_type, llm_config):
        """Create LLM provider."""
        llm_provider_class = ModelMap().get(provider_type)
        assert llm_provider_class is not None, f"Unsupported LLM provider type: {provider_type}"

        print(f"llm_provider_class: {llm_provider_class}")
        llm_provider = globals()[llm_provider_class](llm_config)
        assert llm_provider is not None, f"Unsupported LLM provider type: {provider_type}"
        return llm_provider
    

class LLMOllama(LLMProvider):
    """The Ollama LLM model."""
    def __init__(self, llm_config):
        """Initialize the LLM model."""
        self.type = LLMProviderType.OLLAMA
        self.base_url = llm_config["base_url"]
        self.model = llm_config["model"]
        self.temperature = llm_config["temperature"]
        models_list = self.get_models_list()
        if self.model is None:
            logger.info(f"Setting up default model {models_list[0]}")
            self.llm = ChatOllama(model=models_list[0], base_url=self.base_url, temperature=self.temperature)   
        else:
            logger.info(f"Setting up model from session state for model {self.model}")
            self.llm = ChatOllama(model=self.model, base_url=self.base_url, temperature=self.temperature)

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
        # logger.debug(f"Response: {response}")

        models = response["models"]
        for model in models:
            model_list.append(model["model"])

        return model_list
    
class LLMLlamaCpp(LLMProvider):
    """The Llama.cpp LLM model."""
    def __init__(self, llm_config):
        """Initialize the LLM model."""
        self.type = LLMProviderType.LLAMA_CPP
        self.llm = ChatOpenAI(base_url=llm_config["base_url"], temperature=llm_config["temperature"], api_key=llm_config["api_key"])

class LLMOpenAI(LLMProvider):
    """The Llama.cpp LLM model."""
    def __init__(self, llm_config):
        """Initialize the LLM model."""
        self.type = LLMProviderType.OPENAI
        self.llm = ChatOpenAI(base_url=llm_config["base_url"], temperature=llm_config["temperature"], api_key=llm_config["api_key"])

class LLMNvidia(LLMProvider):
    """The Nvidia LLM model."""
    def __init__(self, llm_config):
        """Initialize the LLM model."""
        os.environ["NVIDIA_API_KEY"] = llm_config["api_key"]
        self.type = LLMProviderType.NVIDIA
        self.llm = ChatNVIDIA(model=llm_config["model"])

    def get_models_list(self):
        return super().get_models_list()


class ModelMap():
    """Map of LLM provider types to their respective classes."""
    def __init__(self):
        self.map = {
            LLMProviderType.OPENAI.value: "LLMOpenAI",
            LLMProviderType.LLAMA_CPP.value: "LLMLlamaCpp",
            LLMProviderType.OLLAMA.value: "LLMOllama",
            LLMProviderType.NVIDIA.value: "LLMNvidia"
        }

    def get(self, provider_type):
        """Get the model class based on the provider type."""
        return self.map.get(provider_type, None)
