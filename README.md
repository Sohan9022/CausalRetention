# 🎯 CausalRetention: Prescriptive Uplift & Retention Decision Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://causal-retention.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB5424?logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://causal-retention.streamlit.app/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)

> **🌐 Live Interactive Platform:** [causal-retention.streamlit.app](https://causal-retention.streamlit.app/)

An industry-grade, prescriptive machine learning platform that moves beyond generic churn prediction to **Causal Uplift Modeling and Financial ROI Optimization**.

Instead of asking *"Who will churn?"*, **CausalRetention** asks:  
> **"Whose churn can actually be prevented by an intervention, and how should a finite marketing budget be allocated to maximize net profit?"**

---

## 📌 Why This Project Stands Out

95% of churn projects simply train a classifier on the Kaggle Telco dataset to predict $P(\text{Churn})$. In enterprise settings (Fintech, Telecom, SaaS, E-Commerce), this leads to **negative ROI** because:
1. **Lost Causes:** Customers who are determined to leave receive discounts and churn anyway (wasted budget).
2. **Sure Things:** Loyal customers who would stay organically receive unnecessary discounts (revenue cannibalization).
3. **Correlation $\neq$ Causation:** Traditional models predict churn risk, not the *treatment effect* of an intervention.

**CausalRetention** solves this using a **Two-Model T-Learner Uplift Architecture** coupled with a **Knapsack Budget Optimizer**, delivering a **board-ready Executive ROI Report**.

---

## 🏗️ Architecture & Pipeline

```
                                  [ Customer Base (Telco Data) ]
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                           [ Treatment Cohort ]    [ Control Cohort ]
                           (Received Discount)      (No Intervention)
                                    │                       │
                                    ▼                       ▼
                              XGBoost Model 1         XGBoost Model 2
                             P(Stay | Offer)        P(Stay | No Offer)
                                    └───────────┬───────────┘
                                                ▼
                                    [ CATE / Uplift Score ]
                                   τ = P(Stay|1) - P(Stay|0)
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                        [ 4 Behavioral Quadrants ]  [ Financial Optimization ]
                        - Persuadables              - Expected Net Profit
                        - Sure Things               - Budget Knapsack Cutoff
                        - Lost Causes               - Diminishing Return Curve
                        - Sleeping Dogs                     │
                                    └───────────┬───────────┘
                                                ▼
                                 [ Streamlit Executive Platform ]
                                 - Interactive Strategy Benchmarks
                                 - Qini / Cumulative Uplift Curves
                                 - CRM Target Dispatch List (CSV)
                                 - Automated HTML/PDF Executive Brief
```

---

## 🧩 The 4 Behavioral Uplift Quadrants

| Quadrant | Profile | Uplift ($\tau$) | Business Action | Financial Impact |
| :--- | :--- | :--- | :--- | :--- |
| 🎯 **Persuadables** | Stay *only* if they receive an incentive | **High Positive** ($\tau \ge 0.10$) | **Target aggressively** | Generates all incremental net profit |
| 🛡️ **Sure Things** | Loyal customers who stay organically | Near Zero ($\tau \approx 0$) | **Suppress discount** | Saves \$10,000s in deadweight loss |
| ⚠️ **Lost Causes** | Dissatisfied users who leave regardless | Near Zero ($\tau \approx 0$) | **Suppress discount** | Prevents wasted subsidy spend |
| 🛑 **Sleeping Dogs** | Dormant accounts triggered by contact | **Negative** ($\tau < 0$) | **Blacklist from campaigns** | Prevents accidental churn |

---

## 📊 Live Benchmark Results ($15,000 Retention Budget)

When evaluating targeting strategies on the exact same customer cohort under a capped **$15,000 budget**:

| Strategy | Budget Spent | Customers Targeted | Customers Saved | Gross Saved | **Net Profit** | **Campaign ROI** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🟢 **Causal Uplift (Our Platform)** | **$14,986.93** | **295** | **+136** | **$127,749.19** | **+$112,762.26** | **+752.4%** |
| 🔴 **Traditional Churn Prediction** | $14,968.19 | 276 | +13 | $13,006.72 | **-$1,961.47** | **-13.1% (Loss)** |
| ⚪ **Random Promotion** | $14,942.02 | 341 | +40 | $23,069.58 | +$8,127.56 | +54.4% |

> **Key Finding:** Standard ML churn prediction produces a **net financial loss (-13.1%)** because it concentrates budget on high-churn-risk accounts who are actually *Lost Causes*. Causal Uplift produces **+752.4% ROI** by isolating *Persuadables*.

---

## 📂 Project Structure

```
causal-retention/
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv       # Official IBM Telco Churn dataset
│   └── processed/
│       └── telco_retention_experiment.csv # A/B retention campaign cohort
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Automated download & A/B trial synthesis
│   ├── feature_pipeline.py                # Scikit-Learn ColumnTransformer pipeline
│   ├── uplift_model.py                    # Two-Model (T-Learner) XGBoost uplift engine
│   ├── optimizer.py                       # 4-quadrant & financial Knapsack allocation
│   └── report_generator.py                # Board-ready HTML/PDF executive brief builder
├── models/
│   └── uplift_model_bundle.joblib         # Serialized models & preprocessors
├── reports/
│   ├── executive_retention_report.html    # Generated executive brief
│   └── target_campaign_roster.csv         # Actionable CRM target list
├── app.py                                 # Interactive Streamlit dashboard
├── run_pipeline.py                        # Standalone end-to-end training & evaluation
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Sohan9022/CausalRetention.git
cd CausalRetention
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
Train the T-Learner models, compute the Qini curves, optimize the retention budget, and generate the executive report in one command:
```bash
python run_pipeline.py
```

### 3. Launch the Interactive Dashboard
Launch the web application with interactive budget sliders, quadrant visualizations, and single-customer simulators (supporting Telecom, Banking, and custom CSV datasets):
```bash
streamlit run app.py
```

### 4. Test on Other Datasets (Cross-Industry Verification)
Run automated end-to-end tests across Banking and B2B SaaS cohorts to verify universal compatibility:
```bash
python test_other_dataset.py
```

---

## 💼 Interview Talking Points

When asked: *"Walk me through your Customer Churn project,"* use this script:

1. **The Problem:** *"Most churn projects only predict binary churn probability, which misguides marketing teams into giving discounts to customers who were either going to stay anyway or were already lost causes."*
2. **The Innovation:** *"I built an Uplift Modeling engine using a Two-Model (T-Learner) XGBoost architecture. It calculates the Conditional Average Treatment Effect (CATE) to isolate persuadable customers."*
3. **The Business Value:** *"I coupled the causal predictions with a Knapsack optimizer under fixed budget constraints. Under a \$15,000 budget, traditional churn targeting resulted in a -\$1,961 net loss, whereas our Causal Uplift platform generated +\$112,762 in net preserved revenue (a 752% ROI)."*
4. **The Deliverable:** *"The project delivers an interactive Streamlit application with live 'what-if' simulation, downloadable CRM target rosters, and an automated board-ready Executive Decision Brief."*

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`), XGBoost
* **Causal Uplift Formulation:** Two-Model T-Learner (CATE estimation, Qini Curves)
* **Optimization:** Knapsack Greedy Profit Maximization
* **UI & Reporting:** Streamlit, Plotly Express, Jinja2/HTML5
