import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator

class TransactionBase(BaseModel):
    merchant: str = Field(..., min_length=1, description="Name of the merchant/store")
    category: str = Field(..., min_length=1, description="Category of spending")
    amount: float = Field(..., gt=0, description="Amount spent, must be greater than zero")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    payment_method: Optional[str] = Field("Unknown", description="UPI, Cards, NetBanking, Cash, etc.")
    confidence: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Confidence of extraction (0 to 1)")
    reference_number: Optional[str] = Field(None, description="Transaction ref or ID")
    raw_text: Optional[str] = None
    ocr_source: Optional[str] = None

    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError("Date must be in YYYY-MM-DD format")
        
        # Verify date parts are valid
        year, month, day = map(int, v.split('-'))
        if not (1 <= month <= 12):
            raise ValueError("Month must be between 01 and 12")
        if not (1 <= day <= 31):
            raise ValueError("Day must be between 01 and 31")
            
        return v

    @field_validator('merchant', 'category')
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        val = v.strip()
        if not val:
            raise ValueError("String field cannot be blank or whitespace-only")
        return val

class TransactionCreate(TransactionBase):
    pass

class TransactionResponse(TransactionBase):
    id: int
    created_at: str

    class Config:
        from_attributes = True

class WeeklySummarySchema(BaseModel):
    start_date: str
    end_date: str
    total_spend: float
    category_breakdown: str  # JSON String
    insights: Optional[str] = None

class MonthlySummarySchema(BaseModel):
    month: str  # YYYY-MM
    total_spend: float
    category_breakdown: str  # JSON String
    insights: Optional[str] = None

class ChatHistorySchema(BaseModel):
    session_id: str
    role: str
    content: str
    timestamp: str

class FinancialInsightSchema(BaseModel):
    type: str
    insight_text: str
    created_at: str
