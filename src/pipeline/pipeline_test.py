import unittest
from src.pipeline.pipeline import Pipeline

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline()
        return super().setUp()
    
if __name__ == "__main__":
    unittest.main()