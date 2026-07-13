# SpendSense 💰 - AI Personal Finance Coach

SpendSense is a production-quality, privacy-first personal finance assistant that operates like ChatGPT. Instead of filling out complex financial spreadsheets or forms, users interact naturally through conversation, receipt uploads, or voice recordings to track their spending habits, identify subscriptions, find anomalies, and receive tailored financial coaching.

---

## 🏗️ System Architecture

SpendSense is built using a decoupled, clean architecture to ensure maintainability, testability, and deterministic financial accounting.

```
                  ┌────────────────────────────────────────┐
                  │           Streamlit Chat UI            │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │        SpendSense Coordinator          │
                  └──────┬────────────┬─────────────┬──────┘
                         │            │             │
        ┌────────────────┘            │             └────────────────┐
        ▼                             ▼                              ▼
┌──────────────┐              ┌──────────────┐               ┌──────────────┐
│ SQLite DB    │              │  ChromaDB    │               │  LLM Service │
│ (SQLAlchemy) │              │  (VectorDB)  │               │  (Agnostic)  │
└──────┬───────┘              └──────┬───────┘               └──────┬───────┘
       │                             │                              │
       ▼                             ▼                              ▼
┌──────────────┐              ┌──────────────┐               ┌──────────────┐
│  Pandas      │              │  Semantic    │               │  Financial   │
│  Analytics   │              │  Retrieval   │               │  Coaching    │
└──────────────┘              └──────────────┘               └──────────────┘
```

### Core Design Principles
1. **SQL for Bookkeeping**: SQLite & SQLAlchemy handle all transaction state, inserts, deletes, and exact numerical aggregations.
2. **Pandas for Analytics**: Python's data analysis libraries (Pandas and NumPy) calculate averages, recurring bill cycles, category growth percentages, and outliers.
3. **ChromaDB for Semantic Context**: ChromaDB indexes weekly/monthly summary insights and advice. The vector store is used only for qualitative semantic retrieval, not raw math.
4. **LLM for Reasoning**: The LLM is used *exclusively* for structured data extraction, conversational understanding, and coaching advice synthesis. It never performs arithmetic.

---

## 🛠️ Features

- **Natural Language Parsing**: Log transactions by simply saying *"I spent ₹350 on lunch at Starbucks yesterday"* or *"Paid ₹850 for electricity bills"*.
- **OCR Receipt Scan**: Upload payment screenshot receipts from Google Pay, PhonePe, or bank SMS alerts. The system extracts transaction details automatically.
- **Voice Transcription**: Speak to the coach using voice uploads (powered by Whisper).
- **Advanced Dashboard**: Interactive Plotly-based spending breakdowns, daily trends, and month-over-month comparisons.
- **Outlier & Subscription Detection**: Detect potential repeating subscriptions and identify spending anomalies.
- **Report Exports**: Download transaction data as CSV, Markdown summaries, or PDF/HTML reports.

---

## 🚀 Installation & Local Setup

### Prerequisites
- Python 3.8+
- ffmpeg (required for Whisper audio decoding)

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/yourusername/spendsense.git
cd spendsense
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```
*Note: By default, the application runs in a **zero-cost mock mode** using offline rule-based transaction extraction if no API keys are provided.*

### 4. Run the Streamlit Interface
```bash
streamlit run app.py
```

### 5. Run the FastAPI Backend Server
```bash
uvicorn src.chat.api:app --reload
```

---

## 🐳 Docker Deployment

To launch SpendSense in an isolated container environment with database persistence:

```bash
docker-compose up --build
```
The Streamlit interface will be available at `http://localhost:8501`.

---

## 🧪 Verification & Testing
To run the complete suite of database, analytics, exports, and RAG vector store mock tests:
```bash
python -m unittest discover -s tests
```
