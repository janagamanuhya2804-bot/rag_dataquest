import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

def transactions_to_df(transactions: List[Any]) -> pd.DataFrame:
    """Convert SQLAlchemy transactions list to a Pandas DataFrame."""
    if not transactions:
        return pd.DataFrame(columns=[
            'id', 'merchant', 'category', 'amount', 'date', 
            'payment_method', 'confidence', 'reference_number', 
            'raw_text', 'ocr_source', 'created_at'
        ])
    
    data = []
    for t in transactions:
        data.append({
            'id': t.id,
            'merchant': t.merchant,
            'category': t.category,
            'amount': t.amount,
            'date': pd.to_datetime(t.date),
            'payment_method': getattr(t, 'payment_method', 'Unknown'),
            'confidence': getattr(t, 'confidence', 1.0),
            'reference_number': getattr(t, 'reference_number', None),
            'raw_text': getattr(t, 'raw_text', None),
            'ocr_source': getattr(t, 'ocr_source', None),
            'created_at': getattr(t, 'created_at', datetime.utcnow().isoformat())
        })
    return pd.DataFrame(data)

class FinanceAnalytics:
    def __init__(self, transactions: List[Any]):
        self.df = transactions_to_df(transactions)
        
    def is_empty(self) -> bool:
        return self.df.empty
        
    def total_spending(self) -> float:
        if self.df.empty:
            return 0.0
        return float(self.df['amount'].sum())

    def average_spend(self) -> float:
        if self.df.empty:
            return 0.0
        return float(self.df['amount'].mean())
        
    def category_spending(self) -> Dict[str, float]:
        if self.df.empty:
            return {}
        summary = self.df.groupby('category')['amount'].sum().to_dict()
        return {k: round(float(v), 2) for k, v in summary.items()}
        
    def merchant_frequency(self) -> Dict[str, int]:
        if self.df.empty:
            return {}
        return self.df['merchant'].value_counts().to_dict()

    def most_frequent_merchant(self) -> str:
        if self.df.empty:
            return "None"
        return str(self.df['merchant'].mode().iloc[0])

    def highest_expense(self) -> Dict[str, Any]:
        if self.df.empty:
            return {"amount": 0.0, "merchant": "None", "category": "None", "date": ""}
        idx = self.df['amount'].idxmax()
        row = self.df.loc[idx]
        return {
            "amount": float(row['amount']),
            "merchant": str(row['merchant']),
            "category": str(row['category']),
            "date": row['date'].strftime('%Y-%m-%d')
        }

    def daily_spending(self) -> Dict[str, float]:
        if self.df.empty:
            return {}
        daily = self.df.groupby(self.df['date'].dt.strftime('%Y-%m-%d'))['amount'].sum().to_dict()
        return {k: round(float(v), 2) for k, v in daily.items()}
        
    def weekly_spending(self) -> Dict[str, float]:
        if self.df.empty:
            return {}
        weekly = self.df.groupby(self.df['date'].dt.to_period('W').astype(str))['amount'].sum().to_dict()
        return {k: round(float(v), 2) for k, v in weekly.items()}
        
    def monthly_spending(self) -> Dict[str, float]:
        if self.df.empty:
            return {}
        monthly = self.df.groupby(self.df['date'].dt.strftime('%Y-%m'))['amount'].sum().to_dict()
        return {k: round(float(v), 2) for k, v in monthly.items()}

    def average_daily_spend(self) -> float:
        if self.df.empty:
            return 0.0
        daily_sums = self.df.groupby(self.df['date'].dt.date)['amount'].sum()
        return float(daily_sums.mean())

    def average_weekly_spend(self) -> float:
        if self.df.empty:
            return 0.0
        weekly_sums = self.df.groupby(self.df['date'].dt.to_period('W'))['amount'].sum()
        return float(weekly_sums.mean())

    def compare_month_vs_last(self) -> Dict[str, Any]:
        """Compares current month spending with the previous month."""
        if self.df.empty:
            return {
                "current_month_name": datetime.now().strftime('%B'),
                "last_month_name": (datetime.now() - timedelta(days=30)).strftime('%B'),
                "current_month_total": 0.0, "last_month_total": 0.0, "diff_amount": 0.0, "diff_percent": 0.0
            }
            
        today = datetime.now()
        cur_month_str = today.strftime('%Y-%m')
        
        first_of_this_month = today.replace(day=1)
        last_day_of_last_month = first_of_this_month - timedelta(days=1)
        last_month_str = last_day_of_last_month.strftime('%Y-%m')
        
        cur_month_df = self.df[self.df['date'].dt.strftime('%Y-%m') == cur_month_str]
        last_month_df = self.df[self.df['date'].dt.strftime('%Y-%m') == last_month_str]
        
        cur_total = float(cur_month_df['amount'].sum())
        last_total = float(last_month_df['amount'].sum())
        
        diff = cur_total - last_total
        percent = 0.0
        if last_total > 0:
            percent = (diff / last_total) * 100
            
        return {
            "current_month_name": today.strftime('%B'),
            "last_month_name": last_day_of_last_month.strftime('%B'),
            "current_month_total": round(cur_total, 2),
            "last_month_total": round(last_total, 2),
            "diff_amount": round(diff, 2),
            "diff_percent": round(percent, 2)
        }

    def category_growth(self) -> Dict[str, Any]:
        """Compares category spending between this month and last month to find fastest increasing category."""
        if self.df.empty:
            return {"fastest_growing_category": "None", "increase_amount": 0.0, "category_trends": {}}
            
        today = datetime.now()
        cur_month_str = today.strftime('%Y-%m')
        
        first_of_this_month = today.replace(day=1)
        last_day_of_last_month = first_of_this_month - timedelta(days=1)
        last_month_str = last_day_of_last_month.strftime('%Y-%m')
        
        # Calculate category totals for both months
        cur_df = self.df[self.df['date'].dt.strftime('%Y-%m') == cur_month_str]
        last_df = self.df[self.df['date'].dt.strftime('%Y-%m') == last_month_str]
        
        cur_cats = cur_df.groupby('category')['amount'].sum()
        last_cats = last_df.groupby('category')['amount'].sum()
        
        # Merge
        merged = pd.DataFrame({'last': last_cats, 'current': cur_cats}).fillna(0)
        merged['diff'] = merged['current'] - merged['last']
        merged['percent_change'] = 0.0
        merged.loc[merged['last'] > 0, 'percent_change'] = (merged['diff'] / merged['last']) * 100
        
        fastest_cat = "None"
        max_increase = 0.0
        
        # Find fastest growing by absolute increase
        positive_growth = merged[merged['diff'] > 0]
        if not positive_growth.empty:
            idx = positive_growth['diff'].idxmax()
            fastest_cat = str(idx)
            max_increase = float(positive_growth.loc[idx, 'diff'])
            
        trends = {}
        for cat, row in merged.iterrows():
            trends[cat] = {
                "last_month": float(row['last']),
                "this_month": float(row['current']),
                "increase": float(row['diff']),
                "percent": float(row['percent_change'])
            }
            
        return {
            "fastest_growing_category": fastest_cat,
            "increase_amount": round(max_increase, 2),
            "category_trends": trends
        }

    def spending_anomalies(self) -> List[Dict[str, Any]]:
        """Detect unusually high transactions (e.g. amount > 2 standard deviations above category average)."""
        if self.df.empty or len(self.df) < 3:
            return []
            
        anomalies = []
        for cat in self.df['category'].unique():
            cat_df = self.df[self.df['category'] == cat]
            if len(cat_df) < 2:
                # Can't calculate std if only one transaction, skip
                continue
                
            mean = cat_df['amount'].mean()
            std = cat_df['amount'].std()
            
            # If standard deviation is very small or zero, set it to mean * 0.5 as threshold
            if pd.isna(std) or std == 0:
                threshold = mean * 2.0
            else:
                threshold = mean + 2 * std
                
            # Filter transactions exceeding threshold
            anom_df = cat_df[cat_df['amount'] > threshold]
            
            for _, row in anom_df.iterrows():
                anomalies.append({
                    "id": int(row['id']),
                    "merchant": str(row['merchant']),
                    "category": str(row['category']),
                    "amount": float(row['amount']),
                    "date": row['date'].strftime('%Y-%m-%d'),
                    "average_for_category": round(float(mean), 2),
                    "deviation_multiplier": round(float((row['amount'] - mean) / std), 2) if std > 0 else 2.0
                })
        return anomalies

    def recurring_merchants(self) -> List[Dict[str, Any]]:
        """Identify merchants with repeating transactions."""
        if self.df.empty:
            return []
            
        freqs = self.df['merchant'].value_counts()
        recurring = []
        
        for merchant, count in freqs.items():
            if count >= 2:
                merchant_df = self.df[self.df['merchant'] == merchant]
                avg_amount = merchant_df['amount'].mean()
                recurring.append({
                    "merchant": str(merchant),
                    "transaction_count": int(count),
                    "average_amount": round(float(avg_amount), 2),
                    "categories": list(merchant_df['category'].unique())
                })
        return recurring

    def potential_subscriptions(self) -> List[Dict[str, Any]]:
        """Identify recurring transactions that occur roughly monthly (same merchant, similar amount)."""
        if self.df.empty:
            return []
            
        subs = []
        # Group transactions by merchant
        for merchant, grp in self.df.groupby('merchant'):
            if len(grp) < 2:
                continue
            
            # Sort by date
            grp = grp.sort_values('date')
            
            # Calculate time diff in days between sequential transactions
            date_diffs = grp['date'].diff().dropna().dt.days.tolist()
            
            # Check if spacing is monthly-like (e.g. 25 to 35 days)
            is_monthly = all(25 <= diff <= 35 for diff in date_diffs) if date_diffs else False
            
            # Or if it belongs to Subscriptions category
            has_subscription_cat = any(grp['category'] == 'Subscriptions')
            
            # Check amount stability (coefficient of variation of amount < 0.1)
            mean_amt = grp['amount'].mean()
            std_amt = grp['amount'].std()
            stable_amt = (std_amt / mean_amt) < 0.15 if not pd.isna(std_amt) and mean_amt > 0 else True
            
            if (is_monthly or has_subscription_cat) and stable_amt:
                subs.append({
                    "merchant": str(merchant),
                    "frequency": "Monthly",
                    "average_amount": round(float(mean_amt), 2),
                    "last_billed": grp['date'].iloc[-1].strftime('%Y-%m-%d'),
                    "category": str(grp['category'].iloc[0])
                })
        return subs

    def budget_suggestions(self) -> List[str]:
        """Generate rules-based suggestions based on spending patterns."""
        if self.df.empty:
            return ["No transaction records. Start adding expenses to receive custom suggestions."]
            
        suggestions = []
        cat_spend = self.category_spending()
        total = self.total_spending()
        
        if total == 0:
            return ["No spending recorded yet."]
            
        # Food suggests
        food_pct = (cat_spend.get("Food", 0) / total) * 100
        if food_pct > 25:
            swiggy_spend = self.df[self.df['merchant'].str.lower().str.contains('swiggy|zomato', na=False)]['amount'].sum()
            if swiggy_spend > 0:
                suggestions.append(
                    f"Your Food spending is high ({food_pct:.1f}% of total). "
                    f"You spent ₹{swiggy_spend:,.2f} on Swiggy/Zomato. "
                    "Try reducing delivery orders by half to save money."
                )
            else:
                suggestions.append(f"Your Food spending is {food_pct:.1f}% of total. Consider cooking more meals at home.")
                
        # Subscriptions suggests
        subs_spend = cat_spend.get("Subscriptions", 0)
        if subs_spend > 1000:
            suggestions.append(
                f"You spend ₹{subs_spend:,.2f} on subscriptions. "
                "Review your streaming or software bills and cancel tools you haven't used this month."
            )
            
        # Comparison suggest
        comp = self.compare_month_vs_last()
        if comp["diff_percent"] > 10:
            suggestions.append(
                f"Your spending is increasing rapidly: this month is {comp['diff_percent']:.1f}% higher than last month. "
                "Consider setting a hard weekly budget constraint."
            )
            
        # Shopping suggest
        shop_pct = (cat_spend.get("Shopping", 0) / total) * 100
        if shop_pct > 20:
            suggestions.append(
                f"Shopping accounts for {shop_pct:.1f}% of your budget. "
                "Try enforcing a 24-hour waiting rule before checking out items on Amazon."
            )
            
        if not suggestions:
            suggestions.append("Great job! Your spending is well distributed across categories. Continue tracking daily.")
            
        return suggestions
