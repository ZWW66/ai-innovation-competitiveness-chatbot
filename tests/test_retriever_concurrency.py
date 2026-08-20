import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from crew import tools


class DetectConcurrentRetriever:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()

    def invoke(self, query):
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.guard:
            self.active -= 1
        return []


class RetrieverConcurrencyTests(unittest.TestCase):
    def test_shared_embedding_retriever_calls_are_serialized(self):
        fake = DetectConcurrentRetriever()
        original = tools._retriever
        tools._retriever = fake
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(tools._retrieve_docs, [f"query-{i}" for i in range(8)]))
        finally:
            tools._retriever = original

        self.assertEqual(fake.max_active, 1)


if __name__ == "__main__":
    unittest.main()
