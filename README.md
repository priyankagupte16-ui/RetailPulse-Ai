# RetailPulse AI 📦

> **Real-World Retail Demand Forecasting and Inventory Optimization Platform**  
> *A Production-Grade Data Science & Machine Learning Portfolio Project*

---

## 📌 Project Overview
RetailPulse AI is an end-to-end data science platform that bridges the gap between predictive machine learning and supply chain decision-making. 

Instead of treating demand forecasting as an isolated algorithmic exercise, RetailPulse AI connects multi-horizon demand forecasting (XGBoost with quantile prediction intervals) directly to an **Inventory Optimization Engine** calculating dynamic Safety Stock ($SS$), Reorder Points ($ROP$), and automated replenishment recommendations.

---

## 🏗️ System Architecture
```
[ PostgreSQL Database ]
        │
        ▼
[ Feature Engineering Pipeline (Lags, Rolling Stats, Promos) ]
        │
        ▼
[ ML Forecasting Engine (XGBoost with P10/P50/P90 Quantiles) ]
        │
        ▼
[ Inventory Optimization Engine (Safety Stock, ROP, EOQ) ]
        │
        ▼
[ FastAPI Gateway (REST Prediction & Risk Endpoints) ]
        │
        ▼
[ Streamlit Analytical Dashboard (Plotly Visualizations) ]
```

---

## 📁 Repository Structure
```
retailpulse-ai/
├── config/
│   └── config.yaml          # Hyperparameters, lag windows, policy settings
├── data/
│   ├── raw/                 # Source retail transactional datasets
│   └── processed/           # Engineered feature snapshots
├── src/
│   ├── health.py            # Diagnostic health check module
│   ├── database/            # Schema, migrations & connection pooling
│   ├── features/            # Feature pipelines (lags, rolling stats)
│   ├── models/              # Model training, inference, and metrics
│   ├── optimization/        # Safety stock & reorder point business logic
│   ├── api/                 # FastAPI REST application
│   └── dashboard/           # Streamlit user interface
├── tests/                   # Pytest test suite
├── .env.example             # Template for environment variables
├── .gitignore               # Ignored environments, caches, and secrets
├── requirements.txt         # Production dependencies
└── README.md                # Documentation & interview guide
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (Python 3.11 / 3.12 / 3.14 recommended)
- Git

### 2. Virtual Environment Setup
```powershell
# Navigate to the project root
cd C:\Users\priya\.gemini\antigravity\scratch\retailpulse-ai

# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run System Diagnostic / Health Check
```powershell
python src/health.py
pytest tests/
```

### 4. Launch the Streamlit Dashboard
```powershell
streamlit run src/dashboard/app.py
```

---

## 🛣️ Development Roadmap
- [x] **Phase 1:** Project Initialization & Environment Setup
- [ ] **Phase 2:** PostgreSQL Database Schema & Ingestion Pipeline
- [ ] **Phase 3:** Feature Engineering & Time-Series Pipeline
- [ ] **Phase 4:** ML Model Training & Walk-Forward Validation
- [ ] **Phase 5:** Inventory Optimization & Replenishment Logic
- [ ] **Phase 6:** FastAPI Backend Service
- [ ] **Phase 7:** Streamlit Analytical Dashboard & Scenario Simulator
- [ ] **Phase 8:** Docker Containerization & Portfolio Polish

---

## 💼 Placement Interview Value
- **No Data Leakage:** Implements strict temporal walk-forward validation instead of random train-test splitting.
- **Quantile Uncertainty:** Quantifies peak demand risk with P90 quantiles rather than simple point predictions.
- **Business ROI:** Translates forecast accuracy directly into stockout reduction and minimized holding costs.
- **Decoupled Architecture:** Clean separation between database, ML pipeline, REST API, and analytical frontend.
