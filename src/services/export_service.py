import csv
import io
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger("spendsense.export")

FPDF_AVAILABLE = False
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    logger.warning("fpdf2/fpdf package not available. PDF export will generate high-fidelity HTML report for browser printing.")

class ExportService:
    def export_to_csv(self, transactions: List[Any]) -> str:
        """Export list of transactions to a CSV string."""
        if not transactions:
            return "ID,Date,Merchant,Category,Amount,Payment Method,Confidence,Reference Number\n"
            
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "ID", "Date", "Merchant", "Category", "Amount", 
            "Payment Method", "Confidence", "Reference Number", "Raw Text"
        ])
        
        # Rows
        for t in transactions:
            writer.writerow([
                t.id,
                t.date,
                t.merchant,
                t.category,
                t.amount,
                getattr(t, 'payment_method', 'Unknown'),
                getattr(t, 'confidence', 1.0),
                getattr(t, 'reference_number', ''),
                t.raw_text or ''
            ])
            
        return output.getvalue()

    def export_to_markdown(self, transactions: List[Any], summaries: List[Dict[str, Any]]) -> str:
        """Export financial history and summaries to a beautiful Markdown report."""
        today_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        md = []
        md.append(f"# SpendSense Personal Finance Report")
        md.append(f"Generated on: {today_str}\n")
        md.append("---")
        
        # 1. Summaries Section
        md.append("## 📊 Financial Summaries\n")
        if not summaries:
            md.append("No summaries generated yet.\n")
        else:
            for s in summaries:
                if "month" in s:
                    md.append(f"### 📅 Monthly Summary: {s['month']}")
                    md.append(f"- **Total Spend:** ₹{s['total_spend']:,.2f}")
                else:
                    md.append(f"### 📅 Weekly Summary: {s['start_date']} to {s['end_date']}")
                    md.append(f"- **Total Spend:** ₹{s['total_spend']:,.2f}")
                
                # Category breakdown
                md.append("\n**Category Breakdown:**")
                breakdown = s.get("breakdown")
                if isinstance(breakdown, str):
                    try:
                        breakdown = json.loads(breakdown)
                    except Exception:
                        pass
                
                if isinstance(breakdown, dict):
                    for cat, amt in breakdown.items():
                        md.append(f"- {cat}: ₹{amt:,.2f}")
                
                md.append(f"\n**Coach Insights:**\n{s.get('insights', 'None')}\n")
                md.append("---")
                
        # 2. Transactions Log Section
        md.append("## 📝 Transaction Log\n")
        if not transactions:
            md.append("No transactions recorded.\n")
        else:
            md.append("| Date | Merchant | Category | Amount | Payment Method | Reference Num |")
            md.append("| --- | --- | --- | --- | --- | --- |")
            for t in transactions:
                pm = getattr(t, 'payment_method', 'Unknown')
                ref = getattr(t, 'reference_number', '-') or '-'
                md.append(f"| {t.date} | {t.merchant} | {t.category} | ₹{t.amount:,.2f} | {pm} | {ref} |")
                
        return "\n".join(md)

    def export_to_pdf_bytes(self, transactions: List[Any], summaries: List[Dict[str, Any]]) -> bytes:
        """
        Generates PDF report. 
        If FPDF is available, compiles programmatic PDF.
        Otherwise, returns bytes of high-fidelity styled HTML for printing.
        """
        if FPDF_AVAILABLE:
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                
                # Title
                pdf.set_font("Arial", 'B', size=16)
                pdf.cell(200, 10, txt="SpendSense Financial Report", ln=1, align='C')
                pdf.set_font("Arial", size=10)
                pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", ln=1, align='C')
                pdf.ln(10)
                
                # Summaries
                pdf.set_font("Arial", 'B', size=14)
                pdf.cell(200, 10, txt="Financial Summaries", ln=1)
                pdf.set_font("Arial", size=11)
                for s in summaries[:3]: # Limit to last 3 for PDF spacing
                    title = f"Summary ({s.get('month', s.get('start_date', ''))})"
                    pdf.set_font("Arial", 'B', size=11)
                    pdf.cell(200, 8, txt=title, ln=1)
                    pdf.set_font("Arial", size=10)
                    pdf.cell(200, 6, txt=f"Total Spend: Rs. {s['total_spend']:,.2f}", ln=1)
                    pdf.ln(2)
                pdf.ln(5)
                
                # Transactions
                pdf.set_font("Arial", 'B', size=14)
                pdf.cell(200, 10, txt="Transactions Log", ln=1)
                pdf.set_font("Arial", size=9)
                
                # Table Headers
                pdf.cell(30, 8, txt="Date", border=1)
                pdf.cell(50, 8, txt="Merchant", border=1)
                pdf.cell(40, 8, txt="Category", border=1)
                pdf.cell(30, 8, txt="Amount", border=1)
                pdf.cell(40, 8, txt="Method", border=1)
                pdf.ln()
                
                for t in transactions[:30]: # Limit to 30 for fit
                    pdf.cell(30, 6, txt=t.date, border=1)
                    pdf.cell(50, 6, txt=t.merchant[:25], border=1)
                    pdf.cell(40, 6, txt=t.category, border=1)
                    pdf.cell(30, 6, txt=f"Rs.{t.amount:.2f}", border=1)
                    pdf.cell(40, 6, txt=getattr(t, 'payment_method', 'Unknown'), border=1)
                    pdf.ln()
                    
                return bytes(pdf.output(dest='S'))
            except Exception as e:
                logger.error(f"Error compiling FPDF: {e}. Falling back to HTML.")
                
        # High fidelity HTML Report bytes as fallback (modern, highly printable)
        today_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #1E1F22; color: #E3E4E6; padding: 30px; }}
                h1, h2, h3 {{ color: #FFFFFF; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 10px; border: 1px solid #35363C; text-align: left; }}
                th {{ background-color: #2B2D31; }}
                tr:nth-child(even) {{ background-color: #242528; }}
                .badge {{ background-color: #00CC96; color: #111214; padding: 3px 8px; border-radius: 4px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>SpendSense Financial Report</h1>
            <p>Generated: {today_str}</p>
            <hr style="border: 1px solid #35363C;">
            
            <h2>📊 Summaries</h2>
        """
        
        for s in summaries:
            period = s.get("month", f"{s.get('start_date')} to {s.get('end_date')}")
            html += f"""
            <div style="margin-bottom: 20px; padding: 15px; background: #2B2D31; border-radius: 8px;">
                <h3>Summary for {period}</h3>
                <p><b>Total Spend:</b> <span class="badge">₹{s['total_spend']:,.2f}</span></p>
                <p><b>Coach Insights:</b> {s.get('insights', '')}</p>
            </div>
            """
            
        html += """
            <h2>📝 Transactions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Merchant</th>
                        <th>Category</th>
                        <th>Amount</th>
                        <th>Method</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for t in transactions:
            pm = getattr(t, 'payment_method', 'Unknown')
            html += f"""
                <tr>
                    <td>{t.date}</td>
                    <td>{t.merchant}</td>
                    <td>{t.category}</td>
                    <td>₹{t.amount:,.2f}</td>
                    <td>{pm}</td>
                </tr>
            """
            
        html += """
                </tbody>
            </table>
        </body>
        </html>
        """
        return html.encode('utf-8')
