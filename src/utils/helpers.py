import os
import logging
from datetime import datetime

def setup_logging(log_level=logging.INFO):
    """Configure logger formatting for application-wide tracing."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def format_currency(amount: float) -> str:
    """Format float amount into standard Indian currency representation."""
    return f"₹{amount:,.2f}"

def parse_date_string(date_str: str) -> datetime:
    """Parse date string YYYY-MM-DD into a datetime object."""
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        return datetime.now()

def get_relative_date_string(days_offset: int) -> str:
    """Get date string relative to today."""
    target_date = datetime.now() + timedelta(days=days_offset)
    return target_date.strftime("%Y-%m-%d")
