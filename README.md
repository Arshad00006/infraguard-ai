# 🛡️ INFRAguard AI

**Explainable AI layer for Infrastructure Project Risk Intelligence**

> Smart India Hackathon 2026 | Team **Sentinel** | Problem Statement **26103**  
> Theme: Smart Automation | Category: Software

**MONITOR → PREDICT → EXPLAIN → ACT**

INFRAguard AI converts existing project-monitoring data (PAIMANA-aligned) into actionable **cost**, **schedule** and **implementation-risk** intelligence with full explainability (SHAP).

---

## ✨ Features

| Tab | What it does |
|-----|----------------|
| **Risk Dashboard** | Portfolio triage – high-risk projects first, sector-wise risk, distribution charts |
| **Project Deep-Dive** | Detailed view of any project with risk gauge and key metrics |
| **Explain Prediction** | SHAP-based explanation – *why* a project is high/low risk |
| **What-If Simulator** | Change parameters (progress, slip, overrun…) and instantly see risk change |
| **Early Warning Queue** | Prioritized intervention list ranked by urgency |

---

## 🏗️ Tech Stack

- **Frontend / App**: Streamlit
- **ML**: CatBoost (tabular classification)
- **Explainability**: SHAP (TreeExplainer)
- **Visualization**: Plotly
- **Data**: Synthetic PAIMANA-aligned portfolio (cost, expenditure, physical progress, milestones, ministry, sector, status…)

---

## 📁 Project Structure

```
infraguard_ai/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── README.md
├── data/
│   └── sample_projects.csv # Generated on first run (or pre-generated)
├── models/                 # Trained CatBoost models (auto-created on first run)
│   ├── cost_risk_model.cbm
│   ├── delay_risk_model.cbm
│   ├── overall_risk_model.cbm
│   └── feature_cols.joblib
├── utils/
│   ├── data_generator.py   # Synthetic PAIMANA-like data generator
│   └── model_utils.py      # Train / predict / SHAP / what-if logic
└── assets/                 # (optional logos / images)
```

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/infraguard-ai.git
cd infraguard-ai

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

On first run the app will:
1. Generate synthetic project data (if `data/sample_projects.csv` does not exist)
2. Train CatBoost risk models and save them under `models/`

Subsequent runs load the saved models instantly.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select your repo → main file `app.py`
4. Deploy.

---

## ☁️ Deploy on Vercel (via Streamlit alternative or static export)

Streamlit is the recommended host. For a pure frontend demo you can also wrap key pages, but full ML + SHAP works best on Streamlit Cloud / Hugging Face Spaces / Render.

---

## 🧠 How it works (Background Logic)

### 1. Data Layer
- Synthetic projects are generated with fields aligned to PAIMANA-style monitoring:
  - Cost (original / revised), expenditure, physical progress %
  - Milestone slip, spend deviation, cost overrun %
  - Ministry, sector, status, planned duration
- Binary risk labels are derived:
  - `cost_risk` = high overrun or extreme spend deviation
  - `delay_risk` = large milestone slip or progress variance
  - `overall_risk` = either of the above

### 2. Model Layer
- Three CatBoost classifiers are trained (cost / delay / overall).
- Time-aware principles and baseline comparison are used in design (documented in PPT).
- Models output risk **probabilities** (0–100%).

### 3. Explain Layer (SHAP)
- TreeExplainer computes feature contributions for any selected project.
- Positive SHAP → feature increases predicted risk.
- Negative SHAP → feature decreases predicted risk.

### 4. Act Layer
- **What-If Simulator**: user changes parameters → model re-predicts → delta shown.
- **Early Warning Queue**: composite urgency score ranks projects for intervention.

---

## 📊 Sample Metrics (Illustrative)

| Model        | Typical AUC (demo) |
|--------------|--------------------|
| Cost Risk    | ~0.72–0.85         |
| Delay Risk   | ~0.70–0.82         |
| Overall Risk | ~0.78–0.88         |

*(Actual numbers depend on the generated data distribution.)*

---

## 📜 License & Notes

- Built for **Smart India Hackathon 2026** demonstration.
- Data is **synthetic** and intended only for prototype / evaluation purposes.
- Not connected to live PAIMANA systems.

---

**Team Sentinel** | SIH 2026 | PS 26103
