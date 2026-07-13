import os
import sys
import tempfile
import streamlit as st
import pandas as pd
from datetime import datetime

# Adjust path to find src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.services.coordinator import SpendSenseCoordinator
from src.analytics.calculator import FinanceAnalytics
from src.analytics.charts import generate_category_pie_chart, generate_daily_trend_chart, generate_comparison_bar_chart
from src.utils.helpers import setup_logging

setup_logging()

# Page config
st.set_page_config(
    page_title="SpendSense - AI Personal Finance Coach",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark CSS styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1E1F22;
        color: #E3E4E6;
    }
    
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* ChatGPT Bubble Styling */
    .chat-bubble {
        padding: 1.2rem 1.6rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        line-height: 1.5;
        max-width: 85%;
    }
    
    .chat-user {
        background-color: #2B2D31;
        color: #E3E4E6;
        margin-left: auto;
        border-bottom-right-radius: 2px;
    }
    
    .chat-assistant {
        background-color: #35363C;
        color: #E3E4E6;
        border-left: 4px solid #00CC96;
        border-bottom-left-radius: 2px;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #111214 !important;
    }
    
    /* Navigation Sidebar items styling */
    div[data-testid="stSidebarUserContent"] .stRadio > label {
        color: #FFFFFF !important;
        font-weight: bold;
    }
    
    .stButton>button {
        background-color: #00CC96 !important;
        color: #111214 !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stButton>button:hover {
        background-color: #00B383 !important;
        color: #111214 !important;
    }
    
    /* Clean download button styling */
    .stDownloadButton>button {
        background-color: #35363C !important;
        color: #FFFFFF !important;
        border: 1px solid #4E5058 !important;
        border-radius: 8px !important;
        font-weight: normal !important;
    }
    
    .stDownloadButton>button:hover {
        background-color: #4E5058 !important;
        border-color: #00CC96 !important;
    }
    
    .stDataFrame {
        background-color: #2B2D31 !important;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session State
if "coordinator" not in st.session_state:
    if not os.environ.get("LLM_PROVIDER"):
        os.environ["LLM_PROVIDER"] = "mock"
    st.session_state.coordinator = SpendSenseCoordinator()

if "session_id" not in st.session_state:
    st.session_state.session_id = "spendsense_default_session"

coordinator = st.session_state.coordinator
session_id = st.session_state.session_id

# Sidebar ONLY contains navigation and configuration
with st.sidebar:
    st.title("SpendSense 💰")
    st.caption("AI Personal Finance Coach")
    st.markdown("---")
    
    # 1. New Chat (Action button)
    if st.button("➕ New Chat", use_container_width=True):
        db = coordinator.get_db_session()
        try:
            from src.database.models import ChatHistory
            db.query(ChatHistory).filter(ChatHistory.session_id == session_id).delete()
            db.commit()
        except Exception as e:
            st.error(f"Error resetting chat: {e}")
        finally:
            db.close()
        st.success("New chat session started.")
        st.session_state.active_navigation = "Chat"
        st.rerun()
        
    st.markdown("---")
    
    # Navigation Radio (Sidebar ONLY contains: New Chat, Transactions, Weekly Summary, Monthly Summary, Analytics, Settings)
    nav_options = {
        "Chat": "💬 Chat Assistant",
        "Transactions": "📝 Transactions",
        "Weekly Summary": "📊 Weekly Summary",
        "Monthly Summary": "📅 Monthly Summary",
        "Analytics": "📈 Analytics Dashboard",
        "Settings": "⚙️ Settings"
    }
    
    if "active_navigation" not in st.session_state:
        st.session_state.active_navigation = "Chat"
        
    selected_nav = st.radio(
        "Navigation",
        options=list(nav_options.keys()),
        format_func=lambda x: nav_options[x],
        label_visibility="collapsed"
    )
    st.session_state.active_navigation = selected_nav
    
    st.markdown("---")
    st.markdown("### 📤 Export Financial Data")
    
    txs = coordinator.get_all_transactions()
    if txs:
        # CSV Export
        csv_data = coordinator.get_csv_export()
        st.download_button(
            label="💾 Download CSV Report",
            data=csv_data,
            file_name="spendsense_transactions.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Markdown Export
        md_data = coordinator.get_markdown_export()
        st.download_button(
            label="📝 Download Markdown Report",
            data=md_data,
            file_name="spendsense_report.md",
            mime="text/markdown",
            use_container_width=True
        )
        
        # PDF/HTML Export
        pdf_bytes = coordinator.get_pdf_export_bytes()
        from src.services.export_service import FPDF_AVAILABLE
        ext = "pdf" if FPDF_AVAILABLE else "html"
        mime = "application/pdf" if FPDF_AVAILABLE else "text/html"
        st.download_button(
            label="📊 Download PDF/HTML Report",
            data=pdf_bytes,
            file_name=f"spendsense_report.{ext}",
            mime=mime,
            use_container_width=True
        )
    else:
        st.caption("No transaction logs available for export.")

# Get global transaction list for main window views
all_txs = coordinator.get_all_transactions()
analytics = FinanceAnalytics(all_txs)

# Main Window Controller
active_tab = st.session_state.active_navigation

if active_tab == "Chat":
    st.header("SpendSense Chat Coach 💬")
    
    # Split layout: Chat history on left, media uploaders on right
    col_chat, col_upload = st.columns([3, 1])
    
    with col_chat:
        chat_history = coordinator.get_chat_history(session_id)
        
        # Scrollable Chat Container
        chat_container = st.container()
        with chat_container:
            if not chat_history:
                st.markdown(
                    '<div class="chat-bubble chat-assistant">'
                    '👋 Welcome to **SpendSense**! I am your personal finance coach.<br><br>'
                    'You can log your spending naturally:<br>'
                    '- <i>"I spent ₹350 on lunch at Starbucks."</i><br>'
                    '- <i>"I paid ₹850 for electricity today."</i><br>'
                    '- <i>"I spent ₹1200 on Amazon."</i><br><br>'
                    'Ask me analytical questions:<br>'
                    '- <i>"How much did I spend on food this month?"</i><br>'
                    '- <i>"Compare this month with last month."</i><br>'
                    '- <i>"Show my biggest purchases."</i><br><br>'
                    'Or upload screenshots of your receipts or record voice notes in the right-hand panel!'
                    '</div>', 
                    unsafe_allow_html=True
                )
            else:
                for message in chat_history:
                    role = message["role"]
                    content = message["content"]
                    
                    if role == "user":
                        st.markdown(f'<div class="chat-bubble chat-user">🧑‍💻 {content}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-bubble chat-assistant">🤖 {content}</div>', unsafe_allow_html=True)
                        
        st.write("")
        user_input = st.chat_input("Ask SpendSense personal finance coach...")
        if user_input:
            with st.spinner("Processing..."):
                coordinator.process_text_message(user_input, session_id)
            st.rerun()

    with col_upload:
        st.markdown("### 📸 OCR Screenshot Scan")
        uploaded_image = st.file_uploader(
            "Upload image (GPay, PhonePe, receipts)", 
            type=["png", "jpg", "jpeg"],
            key="chat_receipt_uploader",
            label_visibility="collapsed"
        )
        if uploaded_image is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_image.name)[1]) as temp_file:
                temp_file.write(uploaded_image.read())
                temp_path = temp_file.name
                
            with st.spinner("Scanning receipt screenshot..."):
                reply = coordinator.process_image_upload(temp_path, session_id)
                st.info(reply)
                
            try:
                os.remove(temp_path)
            except Exception:
                pass
            st.rerun()
            
        st.markdown("---")
        st.markdown("### 🎤 Voice Recording")
        uploaded_audio = st.file_uploader(
            "Upload audio clip (Whisper transcribing)", 
            type=["mp3", "wav", "m4a"],
            key="chat_audio_uploader",
            label_visibility="collapsed"
        )
        if uploaded_audio is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as temp_file:
                temp_file.write(uploaded_audio.read())
                temp_path = temp_file.name
                
            with st.spinner("Transcribing audio voice message..."):
                reply = coordinator.process_voice_recording(temp_path, session_id)
                st.info(reply)
                
            try:
                os.remove(temp_path)
            except Exception:
                pass
            st.rerun()

elif active_tab == "Transactions":
    st.header("Transaction Log 📝")
    st.caption("Review, validate, and manage your raw transaction records.")
    
    if all_txs:
        # Display as Pandas dataframe
        df_display = pd.DataFrame([{
            "ID": t.id,
            "Date": t.date,
            "Merchant": t.merchant,
            "Category": t.category,
            "Amount (₹)": f"₹{t.amount:,.2f}",
            "Method": getattr(t, 'payment_method', 'Unknown'),
            "Confidence": f"{getattr(t, 'confidence', 1.0)*100:.0f}%",
            "Ref Number": getattr(t, 'reference_number', '-') or '-'
        } for t in all_txs])
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Deletion tool
        st.markdown("### 🗑️ Delete Record")
        col_del_id, col_del_btn = st.columns([3, 1])
        with col_del_id:
            del_id = st.selectbox("Select Transaction ID:", options=[t.id for t in all_txs])
        with col_del_btn:
            st.write("") # spacing
            if st.button("Delete Selected", use_container_width=True):
                if coordinator.delete_transaction(del_id):
                    st.success(f"Transaction ID {del_id} deleted successfully.")
                    st.rerun()
                else:
                    st.error("Failed to delete transaction.")
    else:
        st.info("No transactions logged yet. Add expenses in the Chat Assistant view!")

elif active_tab == "WeeklySummary":
    st.header("Weekly Summary 📊")
    st.caption("Generate or view your 7-day spending trends and recommendations.")
    
    if st.button("🔄 Generate New Weekly Summary", use_container_width=True):
        with st.spinner("Compiling summary details..."):
            res = coordinator.generate_periodic_summary("weekly")
            if res["status"] == "success":
                st.success("Weekly Summary compiled successfully! Check details below.")
            else:
                st.error(res["message"])
                
    db = coordinator.get_db_session()
    try:
        from src.database.models import WeeklySummary
        weeks = db.query(WeeklySummary).order_by(WeeklySummary.created_at.desc()).all()
        
        if weeks:
            for w in weeks:
                with st.expander(f"Weekly Summary: {w.start_date} to {w.end_date}", expanded=True):
                    st.markdown(f"**Total Spending:** ₹{w.total_spend:,.2f}")
                    st.markdown("**Category Breakdown:**")
                    try:
                        breakdown = json.loads(w.category_breakdown)
                        for cat, val in breakdown.items():
                            st.write(f"- {cat}: ₹{val:,.2f}")
                    except Exception:
                        st.write(w.category_breakdown)
                    
                    st.markdown(f"**Coach Insights & Suggestions:**\n{w.insights}")
        else:
            st.info("No weekly summaries generated yet. Click the button above to generate.")
    finally:
        db.close()

elif active_tab == "MonthlySummary":
    st.header("Monthly Summary 📅")
    st.caption("Generate or view your monthly financial health summaries.")
    
    if st.button("🔄 Generate New Monthly Summary", use_container_width=True):
        with st.spinner("Compiling monthly health summary..."):
            res = coordinator.generate_periodic_summary("monthly")
            if res["status"] == "success":
                st.success("Monthly Summary compiled successfully!")
            else:
                st.error(res["message"])
                
    db = coordinator.get_db_session()
    try:
        from src.database.models import MonthlySummary
        months = db.query(MonthlySummary).order_by(MonthlySummary.created_at.desc()).all()
        
        if months:
            for m in months:
                with st.expander(f"Monthly Summary: {m.month}", expanded=True):
                    st.markdown(f"**Total Monthly Spend:** ₹{m.total_spend:,.2f}")
                    st.markdown("**Category Breakdown:**")
                    try:
                        breakdown = json.loads(m.category_breakdown)
                        for cat, val in breakdown.items():
                            st.write(f"- {cat}: ₹{val:,.2f}")
                    except Exception:
                        st.write(m.category_breakdown)
                        
                    st.markdown(f"**Coach Insights:**\n{m.insights}")
        else:
            st.info("No monthly summaries generated yet. Click the button above to generate.")
    finally:
        db.close()

elif active_tab == "Analytics":
    st.header("Analytics Dashboard 📈")
    st.caption("Deterministic financial trends, metrics, and anomaly logs.")
    
    if not analytics.is_empty():
        # Top Metrics Row
        col_total, col_avg, col_merchant = st.columns(3)
        with col_total:
            st.metric("Total Spending", f"₹{analytics.total_spending():,.2f}")
        with col_avg:
            st.metric("Average Transaction", f"₹{analytics.average_spend():,.2f}")
        with col_merchant:
            st.metric("Most Frequent Merchant", analytics.most_frequent_merchant())
            
        st.markdown("---")
        
        # Charts Row
        col_pie, col_trend = st.columns(2)
        with col_pie:
            pie_fig = generate_category_pie_chart(analytics.category_spending())
            st.plotly_chart(pie_fig, use_container_width=True)
        with col_trend:
            trend_fig = generate_daily_trend_chart(analytics.daily_spending())
            st.plotly_chart(trend_fig, use_container_width=True)
            
        st.markdown("---")
        
        # Advanced statistics row
        col_growth, col_anom = st.columns(2)
        with col_growth:
            st.markdown("### 📈 Category Growth Trends")
            growth = analytics.category_growth()
            if growth["fastest_growing_category"] != "None":
                st.success(f"Fastest increasing category: **{growth['fastest_growing_category']}** (+₹{growth['increase_amount']:,.2f})")
            else:
                st.info("No positive spend growth detected this month.")
                
            # Table of growth
            growth_df = pd.DataFrame([{
                "Category": cat,
                "Last Month (₹)": f"₹{info['last_month']:,.2f}",
                "This Month (₹)": f"₹{info['this_month']:,.2f}",
                "Increase (₹)": f"₹{info['increase']:,.2f}",
                "Change (%)": f"{info['percent']:.1f}%"
            } for cat, info in growth["category_trends"].items()])
            
            if not growth_df.empty:
                st.dataframe(growth_df, use_container_width=True, hide_index=True)
                
        with col_anom:
            st.markdown("### ⚠️ Spending Anomalies Detected")
            anoms = analytics.spending_anomalies()
            if anoms:
                for a in anoms:
                    st.warning(
                        f"**{a['merchant']}** ({a['category']}): spent **₹{a['amount']:,.2f}** "
                        f"on {a['date']}. (Category average: ₹{a['average_for_category']:,.2f})"
                    )
            else:
                st.info("No unusual spending anomalies detected. Spending is aligned with averages.")
                
        st.markdown("---")
        
        # Recurring and Subscriptions
        col_subs, col_suggs = st.columns(2)
        with col_subs:
            st.markdown("### 📅 Detected Subscriptions / Recurring Bills")
            subs = analytics.potential_subscriptions()
            if subs:
                for s in subs:
                    st.info(
                        f"**{s['merchant']}** ({s['category']}): recurring **₹{s['average_amount']:,.2f}/month** "
                        f"detected. (Last payment: {s['last_billed']})"
                    )
            else:
                st.caption("No monthly recurring subscriptions or billing intervals detected.")
                
        with col_suggs:
            st.markdown("### 💡 Financial Coach Suggestions")
            suggs = analytics.budget_suggestions()
            for s in suggs:
                st.markdown(f"- {s}")
    else:
        st.info("No transaction logs available. Log some expenses to populate analytics!")

elif active_tab == "Settings":
    st.header("LLM Configuration & Credentials ⚙️")
    st.caption("Configure API keys, model parameters, and service adapters.")
    
    provider = st.selectbox(
        "LLM Provider Connection", 
        options=["mock", "openai", "context_dev", "openai_compatible"],
        index=0 if os.environ.get("LLM_PROVIDER") == "mock" else 
              (1 if os.environ.get("LLM_PROVIDER") == "openai" else 
               (2 if os.environ.get("LLM_PROVIDER") == "context_dev" else 3))
    )
    
    api_key = st.text_input(
        "API Key Secret", 
        value=os.environ.get("OPENAI_API_KEY", "") or os.environ.get("CONTEXT_DEV_API_KEY", ""),
        type="password"
    )
    
    model_name = st.text_input(
        "Model Name Identifier", 
        value=os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    )
    
    st.markdown("---")
    st.markdown("### 🎙️ Speech (Whisper STT) Connection")
    whisper_local = st.toggle("Use Local Whisper Model (requires download of base weights)", value=os.environ.get("WHISPER_LOCAL", "false").lower() == "true")
    
    if st.button("💾 Save Settings & Reconnect adapters", use_container_width=True):
        os.environ["LLM_PROVIDER"] = provider
        os.environ["WHISPER_LOCAL"] = "true" if whisper_local else "false"
        
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_MODEL"] = model_name
        elif provider == "context_dev":
            os.environ["CONTEXT_DEV_API_KEY"] = api_key
            os.environ["CONTEXT_DEV_MODEL"] = model_name
        elif provider == "openai_compatible":
            os.environ["OPENAI_COMPATIBLE_API_KEY"] = api_key
            os.environ["OPENAI_COMPATIBLE_API_BASE"] = "https://api.openai.com/v1"
            os.environ["OPENAI_COMPATIBLE_MODEL"] = model_name
            
        # Re-initialize coordinator and reconnect adapters
        st.session_state.coordinator = SpendSenseCoordinator()
        st.success("Settings updated and adapters reconnected successfully!")
        st.rerun()
