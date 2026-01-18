import os
import sys
import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from dotenv import load_dotenv

# --- FIX: Ensure Python can find rag_system.py in the parent folder ---
# This adds the 'multi_tool_agent' folder to your path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now standard import works
from rag_system import get_financial_context

load_dotenv()

# --- DEFINE TOOL ---
def financial_retriever(query: str) -> dict:
    """Uses the local RAG system to find financial news."""
    # Ensure this function name is unique and descriptive
    context = get_financial_context(query)
    return {"status": "success", "data": context}

# --- DEFINE AGENT ---
finance_bot = Agent(
    name="financial_bot",
    # Using the latest recommended model variant for 2026
    model="gemini-3-flash", 
    instruction="You are a financial advisor bot. Use the financial_retriever tool to answer user questions about companies and financial news.",
    # FIX: Function name here must match 'def financial_retriever' above
    tools=[financial_retriever] 
)