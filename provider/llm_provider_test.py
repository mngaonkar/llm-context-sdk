import unittest
from llm_provider import LLMOllama
from loguru import logger

class TestLLMProvider(unittest.TestCase):
    def setUp(self):
        self.provider = LLMOllama("http://127.0.0.1:11434")
        return super().setUp()
    
    def test_get_models_list(self):
        self.provider.get_models_list()

    def test_generate_response(self):
        response = self.provider.generate_response("Best places to visit in california?")
        logger.info(response)

if __name__ == "__main__":
    unittest.main()