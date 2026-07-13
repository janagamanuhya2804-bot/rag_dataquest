import os
import json
import logging
import re
from typing import Dict, Any, List, Optional
import httpx
from datetime import datetime

logger = logging.getLogger("spendsense.llm")

class LLMProvider:
    def generate(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        raise NotImplementedError

class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, api_key: str, api_base: str, model: str):
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        self.model = model

    def generate(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1
        }
        
        if json_mode:
            # Some providers support response_format, some don't. We'll add it
            payload["response_format"] = {"type": "json_object"}
            
        try:
            response = httpx.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling OpenAI-compatible API: {e}")
            raise RuntimeError(f"LLM request failed: {e}")

class MockProvider(LLMProvider):
    """Fallback mock provider using local parsing logic for zero-cost testing."""
    def generate(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        # Find user message
        user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_msg = m["content"]
                break
                
        if json_mode:
            # We are extracting transaction details
            # e.g., "I spent ₹350 on Swiggy for lunch" or "I spent 1200 on Amazon"
            # Try to match amount
            amount_match = re.search(r'(?:₹|Rs\.?|INR)?\s*(\d+(?:\.\d+)?)', user_msg, re.IGNORECASE)
            amount = float(amount_match.group(1)) if amount_match else 0.0
            
            # Try to match merchant
            # E.g. "on Amazon", "on Swiggy", "to Swiggy"
            merchant_match = re.search(r'(?:on|at|to|from|for)\s+([A-Za-z0-9\s]+?)(?:\s+for|\s+on|\s+at|\s+₹|\s*$)', user_msg, re.IGNORECASE)
            merchant = merchant_match.group(1).strip() if merchant_match else "Unknown"
            
            # Categories
            categories = ["Food", "Groceries", "Utilities", "Shopping", "Entertainment", "Transport", "Subscriptions", "Other"]
            category = "Other"
            
            # Look for keyword match
            user_msg_lower = user_msg.lower()
            if any(k in user_msg_lower for k in ["lunch", "dinner", "food", "swiggy", "zomato", "restaurant"]):
                category = "Food"
            elif any(k in user_msg_lower for k in ["electricity", "power", "water", "gas", "bill", "recharge"]):
                category = "Utilities"
            elif any(k in user_msg_lower for k in ["amazon", "flipkart", "myntra", "shopping", "clothes"]):
                category = "Shopping"
            elif any(k in user_msg_lower for k in ["movie", "netflix", "spotify", "prime", "youtube"]):
                category = "Entertainment"
            elif any(k in user_msg_lower for k in ["uber", "ola", "auto", "cab", "metro", "fuel", "petrol"]):
                category = "Transport"
            elif any(k in user_msg_lower for k in ["groceries", "blinkit", "zepto", "instamart", "milk"]):
                category = "Groceries"
                
            today_str = datetime.now().strftime('%Y-%m-%d')
            
            result = {
                "merchant": merchant,
                "category": category,
                "amount": amount,
                "date": today_str
            }
            return json.dumps(result)
        else:
            # We are answering finance coach questions
            # Let's write a simple deterministic parser
            user_msg_lower = user_msg.lower()
            if "how much did i spend on food" in user_msg_lower:
                return "Based on your transaction records, you spent ₹0 on food this month. Add transactions to see analytics."
            elif "compare" in user_msg_lower:
                return "I compared this month with last month. You have no recorded transactions to compare."
            elif "biggest expense" in user_msg_lower:
                return "You have no transactions recorded yet. Please log some expenses first!"
            else:
                return "Hi! I am SpendSense, your AI Personal Finance Coach. You can tell me about your expenses like: 'I spent ₹350 on lunch' or upload a receipt screenshot!"

class LLMService:
    def __init__(self):
        # Load config
        self.provider_name = os.getenv("LLM_PROVIDER", "mock").lower()
        self.provider = self._setup_provider()
        
    def _setup_provider(self) -> LLMProvider:
        if self.provider_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found. Falling back to MockProvider.")
                return MockProvider()
            api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return OpenAICompatibleProvider(api_key, api_base, model)
            
        elif self.provider_name == "context_dev":
            api_key = os.getenv("CONTEXT_DEV_API_KEY")
            if not api_key:
                logger.warning("CONTEXT_DEV_API_KEY not found. Falling back to MockProvider.")
                return MockProvider()
            api_base = os.getenv("CONTEXT_DEV_API_BASE", "https://api.context.dev/v1")
            model = os.getenv("CONTEXT_DEV_MODEL", "gpt-4o-mini")
            return OpenAICompatibleProvider(api_key, api_base, model)
            
        elif self.provider_name == "openai_compatible":
            api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
            api_base = os.getenv("OPENAI_COMPATIBLE_API_BASE")
            model = os.getenv("OPENAI_COMPATIBLE_MODEL", "gpt-4o-mini")
            if not api_key or not api_base:
                logger.warning("OPENAI_COMPATIBLE config incomplete. Falling back to MockProvider.")
                return MockProvider()
            return OpenAICompatibleProvider(api_key, api_base, model)
            
        else:
            return MockProvider()
            
    def generate(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        """Call LLM to generate response."""
        return self.provider.generate(messages, json_mode=json_mode)
