# 📊 PBS What-if Simulator + Vetted Chat

**Deterministic, retrieval-only budget simulator with trust scoring, audit trails, and Streamlit UI**.  

This tool allows analysts to explore the **2024–25 Australian Government Portfolio Budget Statements (PBS)** and procurement data in a controlled, auditable way.  
It combines:  
- **What-if simulation**: project procurement changes into PBS budget impact.  
- **Vetted chatbot**: retrieval-only analyzers with no hallucinations.  
- **Trust scoring**: freshness, coverage, consistency, backtesting.  
- **Audit log**: JSONL file of all chat interactions for compliance.  
- **Evidence tables & provenance**: every answer includes supporting data lineage.  

---

## 🚀 Features

- **PBS Integration**  
  - Auto-downloads the official PBS 2024–25 program expense CSV from [data.gov.au](https://data.gov.au).  
  - Normalizes headers and numbers across encodings.  

- **Procurement Data**  
  - Uses sample procurement dataset (synthetic if none provided).  
  - Daily transactions across ICT, Construction, Consulting, Office Supplies, Travel.  

- **What-if Simulator**  
  - Apply shocks: category-specific changes, inflation, savings.  
  - Compare scenario vs PBS baseline with variance.  

- **Vetted Chatbot (Retrieval-Only)**  
  - Supports vetted queries:  
    - **Budget outlook vs PBS**  
    - **Vendor outliers**  
    - **Procurement red flags**  
  - Rejects unsupported or low-evidence queries.  

- **Trust Scoring**  
  - Composite score based on:  
    - Data freshness  
    - Coverage vs elapsed months  
    - Cross-check consistency  
    - Statistical strength  
    - Forecast backtest accuracy  

- **Audit Trail**  
  - All chat answers logged in `audit_logs_chat.jsonl`.  
  - Includes query, analyzer intent, trust metrics, provenance fingerprints.  

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-org/pbs-what-if-simulator.git
cd pbs-what-if-simulator

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
