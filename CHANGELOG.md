# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-14

### Added
- Complete decoupled modular architecture (`src/` structure).
- SQLite relational database engine with SQLAlchemy models.
- Auto-seeding database logic populating 18 mock transactions spanning the last 45 days.
- Pydantic-validated data transaction extractor adapter.
- Decoupled `FinanceAnalytics` engine using Pandas for calculations, anomaly flags, subscriptions, MoM differences, and category growth.
- Plotly spending charts with dark mode styling matching Streamlit custom dark CSS.
- Speech adapter (Whisper STT) and OCR receipt scans (multimodal GPT-4o-mini & EasyOCR) with mock fallbacks.
- ChromaDB vector store client with custom memory-based TF-IDF fallback for systems with incompatible sqlite3.dll versions.
- Export controllers for downloading CSV, Markdown summaries, and PDF/HTML reports.
- Comprehensive 15-test automated unit test suite.
- Docker configuration (`Dockerfile`, `docker-compose.yml`) for persistent volume deployment.
