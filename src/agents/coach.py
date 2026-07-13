import logging
from typing import List, Dict, Any
from ..services.llm_service import LLMService
from ..analytics.calculator import FinanceAnalytics
from ..rag.vector_db import VectorDBManager

logger = logging.getLogger("spendsense.coach")

class FinanceCoach:
    def __init__(self, llm_service: LLMService, vector_db: VectorDBManager):
        self.llm = llm_service
        self.vector_db = vector_db

    def _determine_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze user query to decide what exact SQL/Pandas calculations to run.
        Returns a dict of intent flags.
        """
        q = query.lower()
        intent = {
            "needs_category_total": False,
            "category": None,
            "needs_comparison": False,
            "needs_biggest_expenses": False,
            "needs_subscriptions": False,
            "needs_anomalies": False,
            "needs_growth": False,
            "needs_general_coaching": False
        }
        
        # Check categories
        categories = ["food", "groceries", "utilities", "shopping", "entertainment", "transport", "subscriptions", "other"]
        for cat in categories:
            if cat in q:
                intent["needs_category_total"] = True
                intent["category"] = cat.capitalize()
                
        if "compare" in q or "last month" in q or "previous month" in q or "growth" in q or "increase" in q:
            intent["needs_comparison"] = True
            intent["needs_growth"] = True
            
        if "biggest" in q or "highest" in q or "max" in q or "expensive" in q or "large purchase" in q:
            intent["needs_biggest_expenses"] = True
            
        if "subscription" in q or "cancel" in q or "recurring" in q:
            intent["needs_subscriptions"] = True
            
        if "anomaly" in q or "unusual" in q or "overspend" in q or "wasting" in q:
            intent["needs_anomalies"] = True
            
        # Default coaching checks
        if not any([intent["needs_category_total"], intent["needs_comparison"], intent["needs_biggest_expenses"], intent["needs_subscriptions"], intent["needs_anomalies"]]):
            intent["needs_general_coaching"] = True
            
        return intent

    def generate_response(self, query: str, transactions: List[Any], session_history: List[Dict[str, str]]) -> str:
        """
        Coaches the user on their personal finances using a hybrid RAG pipeline.
        Determines query intent, executes deterministic calculations on SQLite transactions,
        retrieves semantic context from ChromaDB, and uses the LLM to write the response.
        """
        analytics = FinanceAnalytics(transactions)
        intent = self._determine_query_intent(query)
        
        calculated_facts = []
        
        if analytics.is_empty():
            calculated_facts.append("User has no transaction records in SQLite database yet.")
        else:
            total_spend = analytics.total_spending()
            avg_spend = analytics.average_spend()
            frequent_merchant = analytics.most_frequent_merchant()
            
            calculated_facts.append(f"Total spending overall: ₹{total_spend:,.2f}")
            calculated_facts.append(f"Average transaction amount: ₹{avg_spend:,.2f}")
            calculated_facts.append(f"Most frequent merchant: {frequent_merchant}")
            
            # Category query
            if intent["needs_category_total"] and intent["category"]:
                cat = intent["category"]
                cat_spend = analytics.category_spending().get(cat, 0.0)
                calculated_facts.append(f"Exact spending in '{cat}' category: ₹{cat_spend:,.2f}")
                
            # Month comparison
            if intent["needs_comparison"]:
                comp = analytics.compare_month_vs_last()
                calculated_facts.append(
                    f"Month-over-month comparison:\n"
                    f"- This month ({comp['current_month_name']}) total: ₹{comp['current_month_total']:,.2f}\n"
                    f"- Last month ({comp['last_month_name']}) total: ₹{comp['last_month_total']:,.2f}\n"
                    f"- Difference: ₹{comp['diff_amount']:,.2f} ({comp['diff_percent']}% change)"
                )
                
            # Category Growth
            if intent["needs_growth"] or intent["needs_general_coaching"]:
                growth = analytics.category_growth()
                calculated_facts.append(
                    f"Fastest increasing spend category: {growth['fastest_growing_category']} "
                    f"(Increased by ₹{growth['increase_amount']:,.2f} this month)"
                )
                
            # Biggest expenses
            if intent["needs_biggest_expenses"]:
                highest = analytics.highest_expense()
                calculated_facts.append(
                    f"Highest transaction recorded: ₹{highest['amount']:,.2f} at {highest['merchant']} "
                    f"(Category: {highest['category']}, Date: {highest['date']})"
                )
                # Find top 3 expenses
                df = analytics.df.sort_values(by='amount', ascending=False).head(3)
                top_txs = []
                for _, row in df.iterrows():
                    top_txs.append(f"- ₹{row['amount']:,.2f} at {row['merchant']} on {row['date'].strftime('%Y-%m-%d')} ({row['category']})")
                calculated_facts.append("Top 3 highest transactions:\n" + "\n".join(top_txs))
                
            # Subscriptions
            if intent["needs_subscriptions"]:
                subs = analytics.potential_subscriptions()
                if subs:
                    sub_lines = []
                    for s in subs:
                        sub_lines.append(f"- {s['merchant']} (Average cost: ₹{s['average_amount']:,.2f}, last billed on {s['last_billed']})")
                    calculated_facts.append("Potential subscriptions identified:\n" + "\n".join(sub_lines))
                else:
                    calculated_facts.append("No potential active subscriptions detected based on billing intervals.")

            # Anomalies
            if intent["needs_anomalies"] or intent["needs_general_coaching"]:
                anoms = analytics.spending_anomalies()
                if anoms:
                    anom_lines = []
                    for a in anoms[:2]: # Limit to top 2 for context size
                        anom_lines.append(f"- Spent ₹{a['amount']:,.2f} at {a['merchant']} on {a['date']} (Category avg: ₹{a['average_for_category']:,.2f})")
                    calculated_facts.append("Spending anomalies detected:\n" + "\n".join(anom_lines))
                else:
                    calculated_facts.append("No spending anomalies or unusually large category expenses detected.")

            # General suggestions
            suggestions = analytics.budget_suggestions()
            calculated_facts.append("Rule-based budget suggestions:\n" + "\n".join([f"- {s}" for s in suggestions]))

        # 3. Retrieve relevant records from Vector DB (RAG)
        retrieved_docs = self.vector_db.search(query, n_results=4)
        retrieved_context = []
        for doc in retrieved_docs:
            retrieved_context.append(f"[{doc['metadata'].get('type', 'info')}]: {doc['content']}")
            
        context_str = "\n\n".join(retrieved_context) if retrieved_context else "No prior vector records found."

        # 4. Formulate Prompt for LLM Reasoning
        system_prompt = (
            "You are SpendSense, an expert personal finance coach. You help users manage their money, "
            "track budgets, and spend wisely. Your conversational style is warm, minimal, objective, and "
            "resembles ChatGPT.\n\n"
            "Here are the core rules you MUST follow:\n"
            "1. Use the EXACT calculated figures provided in the facts section. NEVER estimate, invent, or "
            "calculate sums, differences, or averages yourself.\n"
            "2. Be deterministic and explainable. Base your advice strictly on the provided transaction metrics "
            "and retrieved context.\n"
            "3. If the user asks a question that requires numbers, and the number is not in the facts, state that you "
            "do not have that transaction recorded.\n"
            "4. Provide constructive, actionable advice on how to save money, adjust spending habits, or cancel "
            "unnecessary subscriptions when relevant.\n"
            "5. Always use Indian Rupee (₹) as the currency."
        )
        
        prompt_content = (
            f"User Query: \"{query}\"\n\n"
            f"=== DETERMINISTIC FACTS (COMPUTED FROM DATABASE) ===\n"
            + "\n".join(calculated_facts) + "\n\n"
            f"=== SEMANTIC CONTEXT (RETRIEVED FROM VECTOR STORE) ===\n"
            f"{context_str}\n\n"
            f"=== INSTRUCTIONS ===\n"
            f"Answer the user query naturally and helpfully as a personal financial coach. Incorporate the deterministic facts above. "
            f"Do not make up any numbers. Suggest practical coaching tips based on their spending growth, subscriptions, or anomalies."
        )
        
        # Build messages including history
        messages = [{"role": "system", "content": system_prompt}]
        
        for chat in session_history[-6:]:
            messages.append({"role": chat["role"], "content": chat["content"]})
            
        messages.append({"role": "user", "content": prompt_content})
        
        try:
            response = self.llm.generate(messages, json_mode=False)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating coach response: {e}")
            return f"I encountered an error connecting to my reasoning engine. However, based on my local database: " + " ".join(calculated_facts)
