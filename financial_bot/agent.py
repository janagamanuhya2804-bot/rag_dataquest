import os
import sys
import datetime  # Separated from the previous line to fix SyntaxError
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from dotenv import load_dotenv

# Path fix to ensure rag_system.py is found in the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now standard import works
from rag_system import get_financial_context

load_dotenv()

# --- DEFINE THE TOOL FIRST ---
# This fixes the Pylance "reportUndefinedVariable" error
def financial_retriever(query: str) -> dict:
    """Uses the local RAG system to find financial news."""
    context = get_financial_context(query)
    return {"status": "success", "data": context}

# --- DEFINE THE AGENT SECOND ---
finance_bot = Agent(
    name="financial_bot",
    # Using the updated 2026 stable model name
    model="gemini-3-flash-preview", 
    instruction="You are a financial advisor bot. Always use the financial_retriever tool to answer user questions about companies.",
    # Use the name defined in the 'def' block above
    tools=[financial_retriever] 
)