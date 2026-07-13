from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Transaction(Base):
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    merchant = Column(String, nullable=False)
    category = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    date = Column(String, nullable=False)  # YYYY-MM-DD
    payment_method = Column(String, nullable=True, default="Unknown")
    confidence = Column(Float, nullable=True, default=1.0)
    reference_number = Column(String, nullable=True)
    raw_text = Column(Text, nullable=True)
    ocr_source = Column(String, nullable=True)  # file_name, 'chat', or 'speech'
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())

class WeeklySummary(Base):
    __tablename__ = 'weekly_summary'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    start_date = Column(String, nullable=False)  # YYYY-MM-DD
    end_date = Column(String, nullable=False)    # YYYY-MM-DD
    total_spend = Column(Float, nullable=False)
    category_breakdown = Column(Text, nullable=False)  # JSON string
    insights = Column(Text, nullable=True)
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())

class MonthlySummary(Base):
    __tablename__ = 'monthly_summary'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    month = Column(String, nullable=False)  # YYYY-MM
    total_spend = Column(Float, nullable=False)
    category_breakdown = Column(Text, nullable=False)  # JSON string
    insights = Column(Text, nullable=True)
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(String, default=lambda: datetime.utcnow().isoformat())

class FinancialInsight(Base):
    __tablename__ = 'financial_insights'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String, nullable=False)  # 'weekly', 'monthly', 'adhoc'
    insight_text = Column(Text, nullable=False)
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())
