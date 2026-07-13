import os
import sys
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Adjust path to find src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.models import Base, Transaction
from src.models.schemas import TransactionCreate

class TestDatabase(unittest.TestCase):
    def setUp(self):
        # Create an in-memory SQLite database for testing
        self.engine = create_engine("sqlite:///:memory:")
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        Base.metadata.create_all(bind=self.engine)

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(bind=self.engine)

    def test_insert_transaction(self):
        # Create valid transaction via schema
        tx_schema = TransactionCreate(
            merchant="Test Starbucks",
            category="Food",
            amount=350.0,
            date="2026-07-14",
            raw_text="spent 350 at Starbucks",
            ocr_source="chat"
        )
        
        # Save to DB
        db_tx = Transaction(
            merchant=tx_schema.merchant,
            category=tx_schema.category,
            amount=tx_schema.amount,
            date=tx_schema.date,
            raw_text=tx_schema.raw_text,
            ocr_source=tx_schema.ocr_source
        )
        self.session.add(db_tx)
        self.session.commit()
        
        # Query and verify
        saved = self.session.query(Transaction).filter_by(merchant="Test Starbucks").first()
        self.assertIsNotNone(saved)
        self.assertEqual(saved.amount, 350.0)
        self.assertEqual(saved.category, "Food")
        self.assertEqual(saved.date, "2026-07-14")

if __name__ == '__main__':
    unittest.main()
