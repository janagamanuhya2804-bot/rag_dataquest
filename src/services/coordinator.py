import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from ..database.connection import SessionLocal
from ..database.models import Transaction, WeeklySummary, MonthlySummary, ChatHistory, FinancialInsight
from ..models.schemas import TransactionCreate
from .llm_service import LLMService
from ..ocr.ocr_service import OCRService
from ..speech.whisper_service import WhisperService
from .export_service import ExportService
from ..rag.vector_db import VectorDBManager
from ..agents.extractor import TransactionExtractor
from ..agents.coach import FinanceCoach
from ..analytics.calculator import FinanceAnalytics

logger = logging.getLogger("spendsense.coordinator")

class SpendSenseCoordinator:
    def __init__(self):
        # 1. Initialize core infrastructure services
        self.llm = LLMService()
        self.vector_db = VectorDBManager()
        
        # 2. Initialize domain services
        self.ocr = OCRService(self.llm)
        self.speech = WhisperService()
        self.exporter = ExportService()
        
        # 3. Initialize AI agents
        self.extractor = TransactionExtractor(self.llm)
        self.coach = FinanceCoach(self.llm, self.vector_db)
        
        # Initialize DB tables and seed
        from ..database.connection import init_db
        init_db()
        
        # Synchronize SQLite transactions to Vector store if index is empty
        self._sync_vector_db()

    def get_db_session(self) -> Session:
        return SessionLocal()

    def _sync_vector_db(self):
        """Synchronize transactions from SQLite database into ChromaDB vector store if empty."""
        try:
            count = 0
            if getattr(self.vector_db, 'use_chroma', False):
                count = self.vector_db.collection.count()
            else:
                count = len(self.vector_db.fallback_db.ids)
                
            if count == 0:
                logger.info("Vector DB is empty. Syncing transactions from SQLite...")
                all_txs = self.get_all_transactions()
                for tx in all_txs:
                    self.vector_db.add_transaction(
                        tx_id=tx.id,
                        merchant=tx.merchant,
                        category=tx.category,
                        amount=tx.amount,
                        date=tx.date,
                        raw_text=tx.raw_text or ""
                    )
                logger.info(f"Successfully indexed {len(all_txs)} transactions to Vector DB.")
        except Exception as e:
            logger.error(f"Error syncing vector DB: {e}")

    def add_transaction(self, tx_in: Dict[str, Any]) -> Optional[Transaction]:
        """Save transaction to SQLite and index it in ChromaDB."""
        db = self.get_db_session()
        try:
            # Create SQLAlchemy instance
            tx = Transaction(
                merchant=tx_in["merchant"],
                category=tx_in["category"],
                amount=tx_in["amount"],
                date=tx_in["date"],
                payment_method=tx_in.get("payment_method", "Unknown"),
                confidence=tx_in.get("confidence", 1.0),
                reference_number=tx_in.get("reference_number"),
                raw_text=tx_in.get("raw_text"),
                ocr_source=tx_in.get("ocr_source")
            )
            db.add(tx)
            db.commit()
            db.refresh(tx)
            
            # Index transaction in Vector DB (ChromaDB)
            try:
                self.vector_db.add_transaction(
                    tx_id=tx.id,
                    merchant=tx.merchant,
                    category=tx.category,
                    amount=tx.amount,
                    date=tx.date,
                    raw_text=tx.raw_text or ""
                )
            except Exception as ve:
                logger.error(f"Error adding transaction to vector DB: {ve}")
                
            return tx
        except Exception as e:
            logger.error(f"Database transaction insert failed: {e}")
            db.rollback()
            return None
        finally:
            db.close()

    def get_all_transactions(self) -> List[Transaction]:
        db = self.get_db_session()
        try:
            return db.query(Transaction).order_by(Transaction.date.desc()).all()
        finally:
            db.close()

    def delete_transaction(self, tx_id: int) -> bool:
        db = self.get_db_session()
        try:
            tx = db.query(Transaction).filter(Transaction.id == tx_id).first()
            if tx:
                db.delete(tx)
                db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting transaction: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        db = self.get_db_session()
        try:
            chats = db.query(ChatHistory).filter(ChatHistory.session_id == session_id).order_by(ChatHistory.timestamp.asc()).all()
            return [{"role": c.role, "content": c.content} for c in chats]
        finally:
            db.close()

    def save_chat_message(self, session_id: str, role: str, content: str):
        db = self.get_db_session()
        try:
            chat = ChatHistory(session_id=session_id, role=role, content=content)
            db.add(chat)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to save chat message: {e}")
            db.rollback()
        finally:
            db.close()

    def process_text_message(self, text: str, session_id: str) -> str:
        """
        Process chat input from user:
        1. Attempt to extract structured transaction details.
        2. If extracted, save to database and reply with confirmation.
        3. If not, treat as financial question, query databases (RAG), and reply with coaching advice.
        """
        # Save user message
        self.save_chat_message(session_id, "user", text)
        
        # 1. Try transaction extraction
        tx_data = self.extractor.extract(text, source="chat")
        
        if tx_data:
            # Save transaction
            tx = self.add_transaction(tx_data)
            if tx:
                reply = (
                    f"Got it! I've recorded a transaction of **₹{tx.amount:,.2f}** at **{tx.merchant}** "
                    f"under the **{tx.category}** category on **{tx.date}**."
                )
            else:
                reply = "I parsed your transaction details, but failed to save them to the database."
        else:
            # 2. Run Financial Coaching RAG Flow
            # Fetch all transactions from SQLite for exact calculations
            all_txs = self.get_all_transactions()
            # Fetch previous chat history
            history = self.get_chat_history(session_id)
            # Remove last user message from history, as it's passed separately
            if history and history[-1]["role"] == "user":
                history = history[:-1]
                
            reply = self.coach.generate_response(text, all_txs, history)
            
        # Save assistant response
        self.save_chat_message(session_id, "assistant", reply)
        return reply

    def process_image_upload(self, image_path: str, session_id: str) -> str:
        """
        Process image screenshot receipt:
        1. Parse via multimodal LLM or local OCR text.
        2. Extract transaction details and insert.
        """
        # Extract transaction
        result = self.ocr.extract_transaction_multimodal(image_path)
        
        tx_data = None
        if result and not result.get("use_text_pipeline"):
            tx_data = result
        else:
            # Text based OCR pipeline fallback
            raw_text = result.get("raw_text") if result else self.ocr.extract_text_from_image(image_path)
            if raw_text:
                tx_data = self.extractor.extract(raw_text, source=os.path.basename(image_path))
                
        if tx_data:
            # Validate and save transaction
            try:
                tx_data["ocr_source"] = os.path.basename(image_path)
                tx = self.add_transaction(tx_data)
                if tx:
                    reply = (
                        f"Receipt processed! I recorded **₹{tx.amount:,.2f}** spent at **{tx.merchant}** "
                        f"under **{tx.category}** on **{tx.date}**."
                    )
                else:
                    reply = "I extracted receipt details but failed to save them to the database."
            except Exception as e:
                reply = f"I found transaction details in the image but they failed validation: {e}"
        else:
            reply = "Sorry, I couldn't read any transaction details from that screenshot. Make sure the amount and merchant are clearly visible."
            
        user_msg = f"[Uploaded screenshot: {os.path.basename(image_path)}]"
        self.save_chat_message(session_id, "user", user_msg)
        self.save_chat_message(session_id, "assistant", reply)
        return reply

    def process_voice_recording(self, audio_path: str, session_id: str) -> str:
        """Transcribe voice message, then process text transcript."""
        transcript = self.speech.transcribe(audio_path)
        if transcript:
            return self.process_text_message(transcript, session_id)
        else:
            reply = "I couldn't hear or transcribe anything from your voice recording. Please try again."
            self.save_chat_message(session_id, "user", "[Voice message]")
            self.save_chat_message(session_id, "assistant", reply)
            return reply

    def generate_periodic_summary(self, period_type: str = "weekly") -> Dict[str, Any]:
        """
        Calculate total expenses and category breakdowns deterministically,
        then call LLM to generate recommendations.
        Saves to SQLite and indexes summary in ChromaDB.
        """
        all_txs = self.get_all_transactions()
        analytics = FinanceAnalytics(all_txs)
        
        if analytics.is_empty():
            return {"status": "error", "message": "No transaction history to summarize."}
            
        db = self.get_db_session()
        today = datetime.now()
        
        try:
            if period_type == "weekly":
                start_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
                end_date = today.strftime('%Y-%m-%d')
                
                df_week = analytics.df[analytics.df['date'] >= datetime.strptime(start_date, '%Y-%m-%d')]
                weekly_total = float(df_week['amount'].sum())
                
                breakdown = df_week.groupby('category')['amount'].sum().to_dict()
                breakdown_str = json.dumps({k: round(float(v), 2) for k, v in breakdown.items()})
                
                prompt = (
                    f"Write a short, professional bulleted advice block for a personal finance weekly summary.\n"
                    f"Start Date: {start_date}, End Date: {end_date}\n"
                    f"Total Spent: ₹{weekly_total:.2f}\n"
                    f"Breakdown: {breakdown_str}\n"
                    "Provide 2-3 specific action items or recommendations on where the user can save money."
                    "Do NOT calculate or summarize totals. Just write reasoning and advice."
                )
                
                messages = [{"role": "user", "content": prompt}]
                insights = self.llm.generate(messages).strip()
                
                summary = WeeklySummary(
                    start_date=start_date,
                    end_date=end_date,
                    total_spend=weekly_total,
                    category_breakdown=breakdown_str,
                    insights=insights
                )
                db.add(summary)
                db.commit()
                db.refresh(summary)
                
                self.vector_db.add_weekly_summary(
                    summary_id=summary.id,
                    start_date=start_date,
                    end_date=end_date,
                    total_spend=weekly_total,
                    breakdown=breakdown_str,
                    insights=insights
                )
                
                return {
                    "status": "success",
                    "type": "weekly",
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_spend": weekly_total,
                    "breakdown": breakdown,
                    "insights": insights
                }
                
            elif period_type == "monthly":
                month_str = today.strftime('%Y-%m')
                
                df_month = analytics.df[analytics.df['date'].dt.strftime('%Y-%m') == month_str]
                monthly_total = float(df_month['amount'].sum())
                
                breakdown = df_month.groupby('category')['amount'].sum().to_dict()
                breakdown_str = json.dumps({k: round(float(v), 2) for k, v in breakdown.items()})
                
                comp = analytics.compare_month_vs_last()
                
                prompt = (
                    f"Write a short, professional bulleted advice block for a monthly finance summary.\n"
                    f"Month: {month_str}\n"
                    f"Total Spent: ₹{monthly_total:.2f}\n"
                    f"Breakdown: {breakdown_str}\n"
                    f"Comparison: spent {comp['diff_percent']}% change compared to last month.\n"
                    "Provide 2-3 specific suggestions. Focus on saving, budget adjustments, and cancelling unneeded subscriptions."
                    "Do NOT calculate or summarize totals. Just write reasoning and advice."
                )
                
                messages = [{"role": "user", "content": prompt}]
                insights = self.llm.generate(messages).strip()
                
                summary = MonthlySummary(
                    month=month_str,
                    total_spend=monthly_total,
                    category_breakdown=breakdown_str,
                    insights=insights
                )
                db.add(summary)
                db.commit()
                db.refresh(summary)
                
                self.vector_db.add_monthly_summary(
                    summary_id=summary.id,
                    month=month_str,
                    total_spend=monthly_total,
                    breakdown=breakdown_str,
                    insights=insights
                )
                
                return {
                    "status": "success",
                    "type": "monthly",
                    "month": month_str,
                    "total_spend": monthly_total,
                    "breakdown": breakdown,
                    "insights": insights
                }
                
        except Exception as e:
            logger.error(f"Error generating periodic summary: {e}")
            db.rollback()
            return {"status": "error", "message": f"Failed to generate summary: {e}"}
        finally:
            db.close()

    def get_csv_export(self) -> str:
        all_txs = self.get_all_transactions()
        return self.exporter.export_to_csv(all_txs)

    def get_markdown_export(self) -> str:
        all_txs = self.get_all_transactions()
        db = self.get_db_session()
        try:
            weeks = db.query(WeeklySummary).all()
            months = db.query(MonthlySummary).all()
            
            summaries = []
            for w in weeks:
                summaries.append({
                    "start_date": w.start_date,
                    "end_date": w.end_date,
                    "total_spend": w.total_spend,
                    "breakdown": w.category_breakdown,
                    "insights": w.insights
                })
            for m in months:
                summaries.append({
                    "month": m.month,
                    "total_spend": m.total_spend,
                    "breakdown": m.category_breakdown,
                    "insights": m.insights
                })
            return self.exporter.export_to_markdown(all_txs, summaries)
        finally:
            db.close()

    def get_pdf_export_bytes(self) -> bytes:
        all_txs = self.get_all_transactions()
        db = self.get_db_session()
        try:
            weeks = db.query(WeeklySummary).all()
            months = db.query(MonthlySummary).all()
            
            summaries = []
            for w in weeks:
                summaries.append({
                    "start_date": w.start_date,
                    "end_date": w.end_date,
                    "total_spend": w.total_spend,
                    "breakdown": w.category_breakdown,
                    "insights": w.insights
                })
            for m in months:
                summaries.append({
                    "month": m.month,
                    "total_spend": m.total_spend,
                    "breakdown": m.category_breakdown,
                    "insights": m.insights
                })
            return self.exporter.export_to_pdf_bytes(all_txs, summaries)
        finally:
            db.close()
