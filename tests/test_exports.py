import os
import sys
import unittest
from datetime import datetime

# Adjust path to find src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.export_service import ExportService

class MockTransactionObj:
    def __init__(self, id_val, merchant, category, amount, date, payment_method="Unknown"):
        self.id = id_val
        self.merchant = merchant
        self.category = category
        self.amount = amount
        self.date = date
        self.payment_method = payment_method
        self.confidence = 1.0
        self.reference_number = "REF12345"
        self.raw_text = 'spent 350 at Swiggy'
        self.ocr_source = 'chat'
        self.created_at = datetime.utcnow().isoformat()

class TestExports(unittest.TestCase):
    def setUp(self):
        self.exporter = ExportService()
        self.txs = [
            MockTransactionObj(1, "Swiggy", "Food", 350.0, "2026-07-14"),
            MockTransactionObj(2, "Netflix", "Subscriptions", 199.0, "2026-07-14", "Credit Card")
        ]
        self.summaries = [
            {
                "month": "2026-07",
                "total_spend": 549.0,
                "breakdown": {"Food": 350.0, "Subscriptions": 199.0},
                "insights": "Reduce Swiggy delivery. Subscriptions are stable."
            }
        ]

    def test_export_to_csv(self):
        csv_str = self.exporter.export_to_csv(self.txs)
        self.assertIn("Swiggy", csv_str)
        self.assertIn("Netflix", csv_str)
        self.assertIn("REF12345", csv_str)
        # Check header
        self.assertIn("Merchant,Category,Amount", csv_str)

    def test_export_to_markdown(self):
        md_str = self.exporter.export_to_markdown(self.txs, self.summaries)
        self.assertIn("# SpendSense Personal Finance Report", md_str)
        self.assertIn("Monthly Summary: 2026-07", md_str)
        self.assertIn("Swiggy", md_str)
        self.assertIn("Netflix", md_str)

    def test_export_to_pdf_bytes(self):
        pdf_bytes = self.exporter.export_to_pdf_bytes(self.txs, self.summaries)
        self.assertTrue(len(pdf_bytes) > 0)
        # Even if FPDF is not installed, the HTML fallback should be returned as bytes
        self.assertIsInstance(pdf_bytes, bytes)
        # Verify fallback HTML or PDF content exists
        if b"<!DOCTYPE html>" in pdf_bytes or b"<html>" in pdf_bytes:
            self.assertIn(b"SpendSense Financial Report", pdf_bytes)

if __name__ == '__main__':
    unittest.main()
