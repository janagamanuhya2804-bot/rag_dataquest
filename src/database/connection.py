import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from .models import Base, Transaction, WeeklySummary, MonthlySummary

# Database file location
DB_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASE_URL = f"sqlite:///{os.path.join(DB_DIR, 'spendsense.db')}"

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_session = scoped_session(SessionLocal)

def seed_database(db):
    """Seed the database with high-quality sample transactions if empty."""
    # Check if we already have transactions
    if db.query(Transaction).count() > 0:
        return
        
    today = datetime.now()
    
    # 45 days ago range
    dates = {
        "today": today.strftime('%Y-%m-%d'),
        "yesterday": (today - timedelta(days=1)).strftime('%Y-%m-%d'),
        "3_days_ago": (today - timedelta(days=3)).strftime('%Y-%m-%d'),
        "5_days_ago": (today - timedelta(days=5)).strftime('%Y-%m-%d'),
        "7_days_ago": (today - timedelta(days=7)).strftime('%Y-%m-%d'),
        "10_days_ago": (today - timedelta(days=10)).strftime('%Y-%m-%d'),
        "15_days_ago": (today - timedelta(days=15)).strftime('%Y-%m-%d'),
        # Last month bills
        "last_month_1": (today - timedelta(days=28)).strftime('%Y-%m-%d'),
        "last_month_2": (today - timedelta(days=30)).strftime('%Y-%m-%d'),
        "last_month_3": (today - timedelta(days=32)).strftime('%Y-%m-%d'),
        "last_month_4": (today - timedelta(days=35)).strftime('%Y-%m-%d'),
        "last_month_5": (today - timedelta(days=40)).strftime('%Y-%m-%d')
    }
    
    sample_txs = [
        # Subscriptions (stability test)
        Transaction(merchant="Netflix", category="Subscriptions", amount=199.0, date=dates["today"], payment_method="Credit Card", confidence=1.0, raw_text="netflix monthly renewal"),
        Transaction(merchant="Netflix", category="Subscriptions", amount=199.0, date=dates["last_month_2"], payment_method="Credit Card", confidence=1.0, raw_text="netflix monthly renewal"),
        Transaction(merchant="Spotify", category="Subscriptions", amount=119.0, date=dates["3_days_ago"], payment_method="UPI", confidence=1.0, raw_text="spotify premium bill"),
        Transaction(merchant="Spotify", category="Subscriptions", amount=119.0, date=dates["last_month_1"], payment_method="UPI", confidence=1.0, raw_text="spotify premium bill"),
        
        # Utilities
        Transaction(merchant="Electricity Board", category="Utilities", amount=1850.0, date=dates["5_days_ago"], payment_method="UPI", confidence=1.0, raw_text="paid 1850 electricity bill"),
        Transaction(merchant="Electricity Board", category="Utilities", amount=1500.0, date=dates["last_month_3"], payment_method="UPI", confidence=1.0, raw_text="paid 1500 electricity bill"),
        Transaction(merchant="Water Authority", category="Utilities", amount=350.0, date=dates["10_days_ago"], payment_method="Debit Card", confidence=1.0, raw_text="water charge Rs 350"),
        
        # Food & Dining (High Food spend test)
        Transaction(merchant="Swiggy", category="Food", amount=350.0, date=dates["today"], payment_method="UPI", confidence=1.0, raw_text="I spent 350 on lunch at swiggy"),
        Transaction(merchant="Zomato", category="Food", amount=550.0, date=dates["yesterday"], payment_method="UPI", confidence=1.0, raw_text="ordered dinner from zomato 550"),
        Transaction(merchant="Swiggy", category="Food", amount=620.0, date=dates["5_days_ago"], payment_method="UPI", confidence=1.0, raw_text="swiggy order 620"),
        Transaction(merchant="Swiggy", category="Food", amount=280.0, date=dates["last_month_4"], payment_method="UPI", confidence=1.0, raw_text="swiggy order 280"),
        Transaction(merchant="Zomato", category="Food", amount=320.0, date=dates["last_month_5"], payment_method="UPI", confidence=1.0, raw_text="zomato dinner order 320"),
        # Food anomaly (unusually high spend)
        Transaction(merchant="Grand Hyatt Restaurant", category="Food", amount=6500.0, date=dates["7_days_ago"], payment_method="Credit Card", confidence=1.0, raw_text="dinner with friends at Hyatt 6500"),
        
        # Shopping
        Transaction(merchant="Amazon", category="Shopping", amount=1200.0, date=dates["3_days_ago"], payment_method="Credit Card", confidence=1.0, raw_text="amazon shopping cart 1200"),
        Transaction(merchant="Myntra", category="Shopping", amount=2450.0, date=dates["15_days_ago"], payment_method="Credit Card", confidence=1.0, raw_text="bought clothes from Myntra for 2450"),
        Transaction(merchant="Amazon", category="Shopping", amount=890.0, date=dates["last_month_1"], payment_method="Debit Card", confidence=1.0, raw_text="amazon order"),
        
        # Transport
        Transaction(merchant="Uber", category="Transport", amount=280.0, date=dates["yesterday"], payment_method="UPI", confidence=1.0, raw_text="uber ride yesterday"),
        Transaction(merchant="Ola Cabs", category="Transport", amount=340.0, date=dates["10_days_ago"], payment_method="UPI", confidence=1.0, raw_text="ola cab travel 340")
    ]
    
    try:
        db.add_all(sample_txs)
        db.commit()
    except Exception as e:
        print(f"Error seeding database: {e}")
        db.rollback()

def init_db():
    """Initialize database, create tables, and seed sample transactions."""
    Base.metadata.create_all(bind=engine)
    
    # Run seeding
    db = SessionLocal()
    try:
        seed_database(db)
    finally:
        db.close()

def get_db():
    """Dependency for getting db session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
