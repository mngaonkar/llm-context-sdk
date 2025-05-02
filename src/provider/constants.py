from enum import Enum, auto

class LLMProviderType(Enum):
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    OPENAI = "openai"
    NVIDIA = "nvidia"
