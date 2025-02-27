import unittest
from llm_provider import LLMOllama
from loguru import logger
from rich.markdown import Markdown
from rich.console import Console

class TestLLMProvider(unittest.TestCase):
    def setUp(self):
        self.provider = LLMOllama("http://127.0.0.1:11434")
        return super().setUp()
    
    def test_get_models_list(self):
        self.provider.get_models_list()

    def test_generate_response(self):
        console = Console()
        response = self.provider.generate_response("Best places to visit in california?")
        console.print(Markdown(response))

if __name__ == "__main__":
    unittest.main()