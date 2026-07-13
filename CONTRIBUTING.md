# Contributing to SpendSense

Thank you for your interest in contributing to SpendSense! We welcome all contributions to make this personal finance coach even better.

## Development Workflow

### 1. Fork and Clone
Fork the repository on GitHub, then clone your fork locally:
```bash
git clone https://github.com/yourusername/spendsense.git
cd spendsense
```

### 2. Install Dev Dependencies
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Implement Changes
- Adhere to the decoupling design principles (SQL for database bookkeeping, Pandas for analytics calculations, ChromaDB for semantic retrieval, LLM for text coaching reasoning).
- Never use the LLM to perform arithmetic.
- Include docstrings and type hints.

### 4. Run Verification Suite
Ensure all tests pass before proposing a PR:
```bash
python -m unittest discover -s tests
```

### 5. Create Pull Request
Push your changes to your fork and submit a pull request to the `main` branch. Provide a detailed summary of your changes in the description.
