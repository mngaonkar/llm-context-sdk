import unittest
from src.pipeline.pipeline import Pipeline

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline()
        return super().setUp()
    
    def test_setup_pipeline(self):
        self.pipeline.setup()
        assert self.pipeline.llm_provider is not None, "LLM provider should be initialized"
        assert self.pipeline.vector_store is not None, "Vector store should be initialized"
        assert self.pipeline.prompt is not None, "Prompt template should be initialized"
        assert self.pipeline.session_state is not None, "Session state should be initialized"
        print("Pipeline setup test passed.")
    
if __name__ == "__main__":
    unittest.main()
