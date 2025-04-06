import unittest
from src.provider.constants import LLMProviderType
from src.provider.llm_provider_config import LLMProviderConfig
import os
import json

class TestLLMProviderConfig(unittest.TestCase):
    """Test class for LLMProviderConfig."""
    CONFIG_DB_PATH = "./deploy/provider_config"
    CONFIG_DB_NAME = "llm_providers_config.db"

    def setUp(self):
        return super().setUp()
    
    def test_db_initialize(self):
        if os.path.exists(os.path.join(TestLLMProviderConfig.CONFIG_DB_PATH, TestLLMProviderConfig.CONFIG_DB_NAME)):
            os.remove(os.path.join(TestLLMProviderConfig.CONFIG_DB_PATH, TestLLMProviderConfig.CONFIG_DB_NAME))
        self.config = LLMProviderConfig(os.path.join(TestLLMProviderConfig.CONFIG_DB_PATH, TestLLMProviderConfig.CONFIG_DB_NAME))

        for file in os.listdir(TestLLMProviderConfig.CONFIG_DB_PATH):
            if file.endswith(".json"):
                with open(os.path.join(TestLLMProviderConfig.CONFIG_DB_PATH, file), "r") as f:
                    config = json.load(f)
                    print(config)
                    result = self.config.set_llm_provider_config(config["provider_name"], config)
                    assert result == True, f"LLM provider {file} initialization failed"
            
    def test_get_provider_config(self):
        self.config = LLMProviderConfig(os.path.join(TestLLMProviderConfig.CONFIG_DB_PATH, TestLLMProviderConfig.CONFIG_DB_NAME))
        provider_config = self.config.get_llm_provider_config(LLMProviderType.OLLAMA.value)
        assert provider_config is not None, "Provider configuration should not be None"
        print(provider_config)


if __name__ == "__main__":
    unittest.main()
