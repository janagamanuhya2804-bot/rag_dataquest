import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from ..services.llm_service import LLMService
from ..models.schemas import TransactionCreate

logger = logging.getLogger("spendsense.extractor")

class TransactionExtractor:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        
    def extract(self, text: str, source: str = "chat") -> Optional[Dict[str, Any]]:
        """Extract structured transaction details from raw text/OCR using LLM and validate it."""
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        system_prompt = (
            "You are a precise financial data extraction agent.\n"
            "Your goal is to parse the input text (which could be a natural language message, SMS alert, "
            "or OCR text from a screenshot) and extract transaction information.\n\n"
            "You MUST return a JSON object with EXACTLY these fields:\n"
            "- merchant: Name of the vendor, business, or person receiving money (e.g. 'Swiggy', 'Amazon', 'Electricity Board').\n"
            "- category: Group of spending. Select the closest one from: [Food, Groceries, Utilities, Shopping, Entertainment, Transport, Subscriptions, Other].\n"
            "- amount: The numerical money spent (positive float).\n"
            "- date: The date of transaction in 'YYYY-MM-DD' format. If a relative date like 'today' or 'yesterday' is used, compute it.\n"
            f"  If no date is mentioned, use today's date: '{today_str}'.\n"
            "- payment_method: The method used for payment (e.g. 'UPI', 'Credit Card', 'Debit Card', 'Cash', 'Unknown').\n"
            "- confidence: Value between 0.0 and 1.0 indicating your certainty in this extraction.\n"
            "- reference_number: Any reference ID, transaction ID, or check number found in the text. Return null if none exists.\n\n"
            "Rules:\n"
            "1. Output ONLY a valid JSON object. Do not include markdown code blocks, comments, or extra text.\n"
            "2. If multiple transactions are in the text, extract only the first/main one.\n"
            "3. If no transaction is present or the amount cannot be determined, return an empty JSON object {}."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract transaction details from this text:\n\n\"{text}\""}
        ]
        
        try:
            raw_response = self.llm.generate(messages, json_mode=True)
            raw_response = raw_response.strip()
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]
            raw_response = raw_response.strip()
            
            if not raw_response or raw_response == "{}":
                logger.info("No transaction details found in text.")
                return None
                
            parsed = json.loads(raw_response)
            
            if not parsed or "amount" not in parsed or not parsed["amount"]:
                logger.info("Parsed transaction does not contain a valid amount.")
                return None
                
            # Run Pydantic validation (Rule 8: Validate before SQLite insert)
            tx_data = TransactionCreate(
                merchant=parsed.get("merchant", "Unknown"),
                category=parsed.get("category", "Other"),
                amount=float(parsed["amount"]),
                date=parsed.get("date", today_str),
                payment_method=parsed.get("payment_method", "Unknown"),
                confidence=float(parsed.get("confidence", 1.0)),
                reference_number=parsed.get("reference_number"),
                raw_text=text,
                ocr_source=source
            )
            
            return tx_data.model_dump()
            
        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse LLM json response: {je}. Response: {raw_response}")
            return None
        except Exception as e:
            logger.error(f"Error extracting transaction: {e}")
            return None
