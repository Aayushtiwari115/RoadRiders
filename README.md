
# 📊 PBS What-if Simulator + Vetted Chat

## Introduction
The PBS What-if Simulator + Vetted Chat is a professional-grade analytical tool designed to support decision-making in the context of the Australian Government's 2024–25 Portfolio Budget Statements (PBS) and procurement data. Built with Streamlit, it offers a retrieval-only, deterministic analysis environment with trust scoring, audit trails, and a clean, modern user interface.

## Problem Statement
Government analysts and procurement officers often face challenges in interpreting complex budget and procurement datasets. Traditional tools lack interactivity, transparency, and auditability, making it difficult to:
- Forecast budget impacts based on procurement trends.
- Identify anomalies or compliance risks.
- Extract insights using natural language queries.
- Ensure trust and traceability in automated analyses.

## Solution Overview
This simulator addresses these challenges by combining rule-based analytics, NLP-assisted query parsing, and AI-powered fallback mechanisms. It ensures:
- Deterministic, retrieval-only responses with no hallucinations.
- Trust scoring based on data freshness, coverage, consistency, and statistical strength.
- Tamper-evident audit logging for compliance.
- A streamlined UI with chat history and scoped analysis.

## 🚀 Features
### PBS Integration
- Auto-downloads the official PBS 2024–25 program expense CSV from data.gov.au.
- Normalizes headers and financial figures across encodings.

### Procurement Data
- Uses a sample procurement dataset (synthetic if none provided).
- Includes daily transactions across ICT, Construction, Consulting, Office Supplies, and Travel.

### What-if Simulator
- Apply shocks: category-specific changes, inflation, and savings.
- Compare scenario vs PBS baseline with variance analysis.

### Vetted Chat (Retrieval-Only)
- Supports vetted queries:
  - Budget outlook vs PBS
  - Vendor outliers
  - Procurement red flags
- Rejects unsupported or low-evidence queries.

### Trust Scoring
- Composite score based on:
  - Data freshness
  - Coverage vs elapsed months
  - Cross-check consistency
  - Statistical strength
  - Forecast backtest accuracy

### Audit Trail
- All chat answers logged in `audit_logs_chat.jsonl`.
- Includes query, analyzer intent, trust metrics, and provenance fingerprints.

### Evidence Tables & Provenance
- Every answer includes supporting data lineage and evidence tables.

## 🧰 Technologies Used
- Python
- Streamlit
- spaCy (NLP)
- Hugging Face Transformers
- Pandas & NumPy
- Plotly

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/your-org/pbs-what-if-simulator.git
cd pbs-what-if-simulator

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scriptsctivate on Windows

# Install dependencies
pip install -r requirements.txt
```

## ✅ Conclusion
This simulator empowers public sector analysts and decision-makers with a transparent, interactive, and trustworthy tool for budget and procurement analysis. It bridges the gap between structured financial data and intuitive analytics, ensuring auditability and compliance in every interaction.
