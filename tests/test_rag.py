import os
import sys
import unittest

# Adjust path to find src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.vector_db import VectorDBManager

class TestRAG(unittest.TestCase):
    def setUp(self):
        # Initialize Vector DB (it will automatically use ChromaDB or Fallback DB)
        self.vdb = VectorDBManager()

    def test_add_and_search_transaction(self):
        # Index a unique transaction
        self.vdb.add_transaction(
            tx_id=9999,
            merchant="Unique Coffee Shop",
            category="Food",
            amount=150.0,
            date="2026-07-14",
            raw_text="spent 150 on unique coffee shop"
        )
        
        # Search for it
        results = self.vdb.search("coffee shop", n_results=1)
        self.assertTrue(len(results) >= 1)
        
        match = results[0]
        self.assertIn("Unique Coffee Shop", match["content"])
        self.assertEqual(match["metadata"]["category"], "Food")
        self.assertEqual(match["metadata"]["amount"], 150.0)

    def test_add_and_search_summary(self):
        self.vdb.add_monthly_summary(
            summary_id=8888,
            month="2026-06",
            total_spend=5400.0,
            breakdown='{"Utilities": 850, "Food": 1200}',
            insights="Avoid overspending on dinner delivery. Cancel duplicate streaming apps."
        )
        
        # Search for subscription suggestions
        results = self.vdb.search("streaming apps", n_results=1)
        self.assertTrue(len(results) >= 1)
        self.assertIn("dinner delivery", results[0]["content"])
        self.assertIn("streaming apps", results[0]["content"])

if __name__ == '__main__':
    unittest.main()
