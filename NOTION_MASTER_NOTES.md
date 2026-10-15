# 🎯 CausalRetention: Comprehensive Architecture, Mathematical Foundations & Engineering Compendium

> **Author:** Sohan Gadewar  
> **Repository:** [GitHub: Sohan9022/CausalRetention](https://github.com/Sohan9022/CausalRetention)  
> **Interactive Platform:** [causal-retention.streamlit.app](https://causal-retention.streamlit.app/)  
> **Format:** Notion-Ready Master Technical Document  

---

## 📑 Table of Contents
1. [Executive Summary & The Negative ROI Paradox](#1-executive-summary--the-negative-roi-paradox)
2. [Causal Inference & Potential Outcomes Mathematical Framework](#2-causal-inference--potential-outcomes-mathematical-framework)
3. [The 4 Behavioral Uplift Quadrants](#3-the-4-behavioral-uplift-quadrants)
4. [Uplift Evaluation Metrics: Qini & Cumulative Uplift Curves](#4-uplift-evaluation-metrics-qini--cumulative-uplift-curves)
5. [Data Pipeline & Synthetic A/B Experimentation (`src/data_loader.py`)](#5-data-pipeline--synthetic-ab-experimentation-srcdata_loaderpy)
6. [Feature Preprocessing Pipeline (`src/feature_pipeline.py`)](#6-feature-preprocessing-pipeline-srcfeature_pipelinepy)
7. [Two-Model (T-Learner) Machine Learning Engine (`src/uplift_model.py`)](#7-two-model-t-learner-machine-learning-engine-srcuplift_modelpy)
8. [Financial Knapsack Budget Optimizer (`src/optimizer.py`)](#8-financial-knapsack-budget-optimizer-srcoptimizerpy)
9. [Universal Cross-Industry Tabular Pipeline (`src/dynamic_pipeline.py`)](#9-universal-cross-industry-tabular-pipeline-srcdynamic_pipelinepy)
10. [Enterprise Dashboard & What-If Simulator (`app.py`)](#10-enterprise-dashboard--what-if-simulator-apppy)
11. [Automated HTML/PDF Executive Reporting Engine (`src/report_generator.py`)](#11-automated-htmlpdf-executive-reporting-engine-srcreport_generatorpy)
12. [End-to-End Execution, Testing & Docker Deployment](#12-end-to-end-execution-testing--docker-deployment)
13. [Machine Learning Interview & System Defense Masterclass](#13-machine-learning-interview--system-defense-masterclass)

---

# 1. Executive Summary & The Negative ROI Paradox

### 1.1 The Industry Churn Trap
In 95% of machine learning churn projects, the objective is framed as a standard supervised classification problem:
$$\text{Predict } P(\text{Churn}_i = 1 \mid X_i)$$
The business then sorts customer accounts by descending churn risk and distributes retention discounts (e.g., $15–$50 subsidies) to the top decile.

In enterprise production (Telecom, Banking, B2B SaaS, E-Commerce), this traditional strategy routinely results in a **net financial loss**:

```
[ Traditional Supervised Churn Model ]
             │
             ▼ Sort by P(Churn)
Top Churn Risk Accounts:
  ├── ⚠️ Lost Causes (85% of budget): Leave anyway despite receiving discounts ($ wasted)
  └── 🛡️ Sure Things (15% of budget): Would have stayed organically ($ cannibalized)
             │
             ▼
Result: Negative Campaign ROI (-13.1% in our benchmark)
```

### 1.2 The Causal Paradigm Shift
CausalRetention shifts the objective from **prediction** to **prescription**:

| Dimension | Traditional Churn Prediction | CausalRetention (Prescriptive Uplift) |
| :--- | :--- | :--- |
| **Core Question** | *"Who is most likely to churn?"* | *"Whose retention can actually be influenced by an offer?"* |
| **Statistical Goal** | Model $\mathbb{E}[Y \mid X]$ | Model $\mathbb{E}[Y(1) - Y(0) \mid X]$ |
| **Target Audience** | Highest baseline churn probability | High positive treatment effect (Persuadables) |
| **Capital Allocation** | Unconstrained risk sorting | Knapsack Profit Maximization under Budget $B$ |
| **Financial Outcome** | **-$1,961 Loss (-13.1% ROI)** | **+$112,762 Profit (+752.4% ROI)** |

### 1.3 Empirical Benchmark ($15,000 Budget Allocation)
Benchmarking targeting strategies on the exact same 7,043 customer cohort under an identical $15,000 budget cap:

```
+-----------------------------------+---------------+------------------+-----------------+--------------------+--------------------+---------------+
| Strategy                          | Budget Spent  | Accounts Targeted| Customers Saved | Gross Value Saved  | Net Profit Gained  | Campaign ROI  |
+-----------------------------------+---------------+------------------+-----------------+--------------------+--------------------+---------------+
| 🟢 Causal Uplift (Our Platform)   | $14,986.93    | 295              | +136            | $127,749.19        | +$112,762.26       | +752.4%       |
| 🔴 Traditional Churn Prediction   | $14,968.19    | 276              | +13             | $13,006.72         | -$1,961.47         | -13.1% (Loss) |
| ⚪ Random Promotion (Baseline)    | $14,942.02    | 341              | +40             | $23,069.58         | +$8,127.56         | +54.4%        |
+-----------------------------------+---------------+------------------+-----------------+--------------------+--------------------+---------------+
```

> **Why did Random outperform Traditional Churn?**  
> Traditional churn prediction concentrates 100% of its budget on high-churn-risk accounts, which are heavily saturated with Lost Causes. Random promotion disperses incentives across the general population, capturing a few Persuadables by chance, whereas traditional churn systematically avoids them.

---

# 2. Causal Inference & Potential Outcomes Mathematical Framework

### 2.1 Rubin Causal Model (Potential Outcomes)
For each customer $i$ in cohort size $N$:
* $T_i \in \{0, 1\}$: Binary treatment indicator ($1 = \text{Retention Offer Sent}$, $0 = \text{Control / No Contact}$).
* $Y_i(1) \in \{0, 1\}$: Potential retention outcome if customer $i$ receives the treatment.
* $Y_i(0) \in \{0, 1\}$: Potential retention outcome if customer $i$ does NOT receive the treatment.

The observed retention outcome $Y_i$ is expressed as:
$$Y_i = T_i Y_i(1) + (1 - T_i) Y_i(0)$$

### 2.2 The Fundamental Problem of Causal Inference
For any individual customer $i$, we observe only the factual outcome corresponding to the assigned treatment:
$$\text{If } T_i = 1 \implies \text{Observe } Y_i(1), \quad Y_i(0) \text{ is counterfactual}$$
$$\text{If } T_i = 0 \implies \text{Observe } Y_i(0), \quad Y_i(1) \text{ is counterfactual}$$
Because individual counterfactuals are unobservable, individual causal effects cannot be measured directly.

### 2.3 Individual Treatment Effect (ITE) & CATE
We define the Individual Treatment Effect as:
$$\tau_i = Y_i(1) - Y_i(0)$$

To operationalize this over customer feature vectors $X_i \in \mathbb{R}^d$, we estimate the **Conditional Average Treatment Effect (CATE)**, denoted as the Uplift Score $\tau(X)$:
$$\tau(X) = \mathbb{E}[Y_i(1) - Y_i(0) \mid X = x]$$
$$\tau(X) = P(Y=1 \mid X=x, T=1) - P(Y=1 \mid X=x, T=0)$$

### 2.4 Unconfoundedness and Overlap Assumptions
For CATE to be identifiable from observational or experimental data:
1. **Unconfoundedness (Conditional Independence Assumption):**
   $$(Y(1), Y(0)) \perp T \mid X$$
   *In randomized A/B trials, this holds unconditionally: $(Y(1), Y(0)) \perp T$.*
2. **Positivity (Overlap):**
   $$\forall x \in \text{supp}(X), \quad 0 < P(T = 1 \mid X = x) < 1$$
   *Every customer profile has a non-zero probability of receiving either treatment or control.*
3. **SUTVA (Stable Unit Treatment Value Assumption):**
   *The treatment applied to customer $i$ does not affect the retention outcome of customer $j$ (no network interference), and there are no hidden variations of the treatment.*

---

# 3. The 4 Behavioral Uplift Quadrants

The customer base decomposes into 4 distinct behavioral archetypes based on their counterfactual retention pairs $(Y_i(0), Y_i(1))$:

```
                            [ Counterfactual State Space ]
                                    Y(1) = Stay
                                         ▲
                                         │
                   🛡️ SURE THINGS        │        🎯 PERSUADABLES
                  Y(0) = 1, Y(1) = 1     │       Y(0) = 0, Y(1) = 1
                  Organic Retention      │       Incentive-Driven Stay
                  Uplift τ ≈ 0           │       Uplift τ >> 0
                  Action: SUPPRESS       │       Action: TARGET AGGRESSIVELY
                                         │
        Y(0) = Leave ────────────────────┼────────────────────► Y(0) = Stay
                                         │
                   ⚠️ LOST CAUSES        │        🛑 SLEEPING DOGS
                  Y(0) = 0, Y(1) = 0     │       Y(0) = 1, Y(1) = 0
                  Irreversible Churn     │       Contact-Induced Churn
                  Uplift τ ≈ 0           │       Uplift τ < 0
                  Action: SUPPRESS       │       Action: DO NOT CONTACT
                                         │
                                         ▼
                                   Y(1) = Leave
```

### Detailed Breakdown:

#### 1. 🎯 Persuadables ($Y(0)=0, Y(1)=1$)
* **Characteristics:** Moderate tenure, month-to-month or 1-year contracts, price-sensitive, active users.
* **Uplift Effect:** $\tau(X) \ge +0.10$ (often $+0.30$ to $+0.45$).
* **Business Prescription:** **Target Aggressively.**
* **Financial Value:** This is the only cohort that produces incremental retained revenue.

#### 2. 🛡️ Sure Things ($Y(0)=1, Y(1)=1$)
* **Characteristics:** Long tenure (>36 months), multi-year contracts, multiple bundled lines, high organic loyalty.
* **Uplift Effect:** $\tau(X) \approx 0$ ($P(\text{Stay} \mid T=0) \ge 0.85$).
* **Business Prescription:** **Suppress Intervention / Hold Out.**
* **Financial Value:** Suppressing discounts saves $10,000s in deadweight subsidies without impacting retention.

#### 3. ⚠️ Lost Causes ($Y(0)=0, Y(1)=0$)
* **Characteristics:** Extreme friction signals, low tenure, high charges, frequent technical complaints, competitors entered market.
* **Uplift Effect:** $\tau(X) \approx 0$ ($P(\text{Stay} \mid T=0) < 0.20$ and $P(\text{Stay} \mid T=1) < 0.25$).
* **Business Prescription:** **Suppress Intervention.**
* **Financial Value:** Prevents capital dissipation on customers who redeem subsidies and cancel anyway.

#### 4. 🛑 Sleeping Dogs ($Y(0)=1, Y(1)=0$)
* **Characteristics:** Low utilization, auto-pay enabled, low ticket size, quiet accounts.
* **Uplift Effect:** $\tau(X) < -0.05$ (Negative treatment effect).
* **Business Prescription:** **Blacklist / Strict Do Not Contact.**
* **Financial Value:** Contacting these accounts reminds them they are paying for an unused subscription, prompting immediate cancellation.

---

# 4. Uplift Evaluation Metrics: Qini & Cumulative Uplift Curves

### 4.1 Why Standard ML Metrics Fail
* **ROC-AUC & Log-Loss:** Measure how well a model predicts the factual label $Y$. They cannot evaluate whether an individual's outcome changed due to an intervention.
* **Counterfactual Inobservability:** Because individual $\tau_i$ is never directly observed, we cannot calculate RMSE against a true $\tau_i$.

### 4.2 Decile-Based Cumulative Uplift Metric
To evaluate causal performance, we rank a holdout test set descending by predicted uplift $\hat{\tau}(X)$ and partition it into deciles $k \in \{1, 2, \dots, 10\}$:

$$Q(k) = Y_{T, k} - \left( Y_{C, k} \times \frac{N_{T, k}}{N_{C, k}} \right)$$

Where:
* $N_{T, k}$: Cumulative number of treated customers up to decile $k$.
* $N_{C, k}$: Cumulative number of control customers up to decile $k$.
* $Y_{T, k}$: Cumulative retained customers in the treated group up to decile $k$.
* $Y_{C, k}$: Cumulative retained customers in the control group up to decile $k$.
* $\frac{N_{T, k}}{N_{C, k}}$: Normalization factor that balances treatment vs control sample sizes.

### 4.3 Qini Curve Interpretation
```
Cumulative Incremental Retained Customers
       ▲
       │                  ┌────────────── Optimal Uplift Model
       │                .·   (Steep initial gain: isolates Persuadables)
       │              .·
       │            .·
       │          .·
       │        .·
       │      .·
       │    .·
       │  .·───────────────────────────── Random Targeting Line
       │.·
       └───────────────────────────────────► Deciles / % Customers Targeted
       0%     20%     40%     60%    100%
```

* **Steep Early Slope:** A superior causal model exhibits a steep rise in the first 20–30% of accounts (capturing all Persuadables).
* **Plateau / Downward Bend:** When the curve reaches Sure Things and Lost Causes ($\tau \approx 0$), the line levels off. If it dips downward, the model is targeting Sleeping Dogs ($\tau < 0$).
* **Qini Score (AUUC):** The area between the model's cumulative uplift curve and the random diagonal baseline.

---

# 5. Data Pipeline & Synthetic A/B Experimentation (`src/data_loader.py`)

### 5.1 Dataset Schema & Cleaning Decisions
* **Raw Dataset:** Official IBM Telco Customer Churn (7,043 rows, 21 columns).
* **Data Cleansing Rules:**
  1. `TotalCharges` contains whitespace strings (`" "`) for 11 accounts where `tenure == 0`. These are cleanly coerced to float `0.0`.
  2. Binary integer flags (e.g., `SeniorCitizen` $0/1$) are converted to string labels (`"No"`, `"Yes"`) for consistent one-hot encoding.

### 5.2 Unit Economics Formulation
To calculate financial returns, real-world revenue and cost parameters are established:
$$\text{annual\_clv} = \text{MonthlyCharges} \times 12$$
$$\text{offer\_cost} = \$15.00 + (\text{MonthlyCharges} \times 0.15 \times 3)$$
* \$15.00 represents administrative, mailing, and CRM outbound operational expense.
* The variable component provides a 15% discount on the customer's monthly bill for a 3-month commitment window.

### 5.3 Synthetic Experiment Generative Mechanism
In production, models train on randomized A/B trials. `generate_retention_experiment()` simulates this:

```python
# 1. Randomized A/B Assignment
treatment = np.random.binomial(1, 0.5, size=n)

# 2. Domain Risk Scoring
baseline_score = (
    0.35 * is_m2m +
    0.20 * tenure_norm +
    0.15 * high_charges +
    0.15 * has_no_support +
    0.10 * is_fiber +
    0.05 * electronic_check
)

# 3. Behavioral Archetype Assignment
# A. Sure Things (2-year contract or tenure > 40)
p_ctrl = uniform(0.90, 0.99), p_treat = uniform(0.92, 0.99)

# B. Lost Causes (baseline_score > 0.75)
p_ctrl = uniform(0.05, 0.20), p_treat = uniform(0.10, 0.25)

# C. Sleeping Dogs (autopay, paperless, monthly charges < $35, no tech support)
p_ctrl = 0.92, p_treat = 0.78  # Contacting triggers cancellation!

# D. Persuadables (moderate risk)
p_ctrl = uniform(0.30, 0.55), p_treat = p_ctrl + uniform(0.30, 0.45)
```

The observed outcome is sampled from the realized branch:
$$Y_i \sim \text{Bernoulli}(T_i \cdot P_i(\text{Treat}) + (1 - T_i) \cdot P_i(\text{Ctrl}))$$

---

# 6. Feature Preprocessing Pipeline (`src/feature_pipeline.py`)

### 6.1 ColumnTransformer Architecture
Production pipelines must prevent feature leakage between train and test splits, and maintain column name lineage for tree interpretability:

```python
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ],
    remainder="drop"
)
```

### 6.2 Key Production Features:
* `sparse_output=False`: Ensures dense matrix generation, compatible with XGBoost array ingestion.
* `handle_unknown="ignore"`: Guards against runtime exceptions when unobserved categorical levels appear during inference.
* `get_feature_names()`: Extracts human-readable encoded names (`Contract_Two year`, `TechSupport_No internet service`) for feature importance mapping.

---

# 7. Two-Model (T-Learner) Machine Learning Engine (`src/uplift_model.py`)

### 7.1 Meta-Learner Architectural Trade-offs

```
                  ┌──────────────────────────────────────────────┐
                  │          Causal Meta-Learner Taxonomy        │
                  └──────────────────────────────────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
    [ S-Learner ]                   [ T-Learner ]                   [ X-Learner ]
• Single model M(X, T)          • Dual models M₁(X), M₀(X)      • Two-stage imputation
• Prone to treatment dilution   • Unbiased subgroup learning    • Specialized for severe
• T regularized to zero         • Optimal for balanced A/B       imbalance (e.g. 99/1 split)
```

#### Why T-Learner is Selected:
In tree ensembles, features with high split variance (e.g., `tenure`, `TotalCharges`) dominate early nodes. In an S-Learner, the treatment flag $T$ often gets regularized out, yielding near-zero uplift for all customers.  
By splitting the dataset into treatment cohort $D_1$ and control cohort $D_0$, the T-Learner trains two independent XGBoost models:
$$M_1(X) = \hat{P}(Y=1 \mid X, T=1) \quad \text{trained on } D_1$$
$$M_0(X) = \hat{P}(Y=1 \mid X, T=0) \quad \text{trained on } D_0$$
$$\hat{\tau}(X) = M_1(X) - M_0(X)$$

### 7.2 XGBoost Hyperparameters & Regularization
```python
XGBClassifier(
    n_estimators=120,
    max_depth=4,            # Shallow depth restricts spurious subgroup interaction
    learning_rate=0.05,     # Conservative shrinkage
    subsample=0.8,          # Row subsampling for variance reduction
    colsample_bytree=0.8,   # Feature subsampling to prevent single-feature dominance
    eval_metric="logloss",
    random_state=42
)
```

### 7.3 Model Artifact Persistence
The bundle is serialized via `joblib` into `models/uplift_model_bundle.joblib`:
```python
bundle = {
    "preprocessor": self.preprocessor,
    "model_treatment": self.model_treatment,
    "model_control": self.model_control,
    "model_baseline_churn": self.model_baseline_churn,
    "feature_names": self.feature_names,
    "random_state": self.random_state
}
```

---

# 8. Financial Knapsack Budget Optimizer (`src/optimizer.py`)

### 8.1 Financial Unit Formulas
For every customer $i$:
1. **Expected Gross Revenue Saved:**
   $$\text{GrossSaved}_i = \hat{\tau}_i \times \text{CLV}_i$$
2. **Expected Net Profit:**
   $$\text{NetProfit}_i = (\hat{\tau}_i \times \text{CLV}_i) - \text{OfferCost}_i$$
3. **Capital Efficiency (Profit per Dollar Spent):**
   $$\text{EfficiencyRatio}_i = \frac{\text{NetProfit}_i}{\text{OfferCost}_i}$$

### 8.2 Why Positive Uplift Does Not Guarantee Profit
Suppose a customer has an uplift score of $\tau = +0.06$ (+6% retention boost):
* If $\text{MonthlyCharges} = \$30 \implies \text{Annual CLV} = \$360$.
* Expected Gross Saved = $0.06 \times \$360 = \$21.60$.
* Offer Cost = $\$15 + (30 \times 0.15 \times 3) = \$28.50$.
* **Expected Net Profit:** $\$21.60 - \$28.50 = \mathbf{-\$6.90}$ (Net Financial Loss).

> 💡 **Takeaway:** Targeting every customer with $\tau > 0$ loses money. Retention capital should only be spent when $\tau \cdot \text{CLV} > \text{Cost}$.

### 8.3 0/1 Knapsack Greedy Allocation Algorithm
Given a fixed retention budget cap $B$ (e.g., $15,000):

$$\max \sum_{i \in \text{Targeted}} \text{NetProfit}_i \quad \text{subject to} \quad \sum_{i \in \text{Targeted}} \text{OfferCost}_i \le B$$

```
Algorithm: Greedy Knapsack Allocation
Input: Scored DataFrame df, Budget B, Threshold tau_min
1. Filter: Candidate Pool = { i ∈ df | tau_i >= tau_min AND NetProfit_i > 0 }
2. Sort Candidate Pool descending by expected_net_profit.
3. Compute cumulative expenditure:
      CumCost_k = sum_{j=1}^k OfferCost_j
4. Find index cutoff k* = max { k | CumCost_k <= B }
5. Targeted Roster = { 1, 2, ..., k* }
6. Holdout Pool = All remaining customers (No offer sent)
```

---

# 9. Universal Cross-Industry Tabular Pipeline (`src/dynamic_pipeline.py`)

`src/dynamic_pipeline.py` enables zero-code ingestion of arbitrary tabular datasets from Banking, B2B SaaS, and Kaggle.

```
                  Arbitrary External CSV
                            │
                            ▼
           [ Dynamic Schema Inference Engine ]
    ├── Customer ID Detection (Regex on ID, clientnum, user_id)
    ├── Target Standardization (Churn / Exited / Attrited -> 0/1)
    ├── Randomized Pilot Synthesis (if no treatment column exists)
    └── CLV Proxy Discovery (Spend, Charges, or Banking Net Interest Margin)
                            │
                            ▼
          [ Dynamic Scikit-Learn Imputer & Encoder ]
    ├── SimpleImputer(strategy='median') for numerics
    └── SimpleImputer + OneHotEncoder(max_categories=15) for categories
                            │
                            ▼
            [ UniversalTLearner (XGBoost) ]
```

### 9.1 Intelligent CLV Proxies
* **SaaS / Telecom:** Detects columns like `mrr`, `charges`, `revenue`, `spend`.
  $$\text{CLV} = \text{MonthlySpend} \times 12$$
* **Retail Banking:** Detects `balance` or `salary`.
  $$\text{CLV} = \text{Deposit Balance} \times 0.04 \text{ (Net Interest Margin Proxy)}$$
  $$\text{CLV} = \text{Estimated Salary} \times 0.015 \text{ (Discretionary Financial Spend Proxy)}$$

---

# 10. Enterprise Dashboard & What-If Simulator (`app.py`)

### 10.1 Design System & Typography
* Dark-canvas design inspired by Linear, Supabase, and Vercel.
* Background gradient: `#111827` to `#080C14`.
* Typography: `Plus Jakarta Sans` for UI, `JetBrains Mono` for numerical values and code.

### 10.2 The 5 Core Workspaces (Tabs)
1. **📊 Strategy Comparison:**
   * Executive KPI cards (Budget Utilized, Net Profit Gained, Net Preserved Customers, Campaign ROI).
   * 3-Way Strategy Benchmark Table.
   * Plotly interactive Qini decile curves and Cumulative Uplift gains.
   * Diminishing Marginal Returns Curve (Profit vs. Budget).
2. **🧩 Behavioral Cohorts:**
   * 2D Scatter Plot ($P(\text{Stay} \mid \text{Control})$ vs Uplift $\tau(X)$).
   * Color-coded quadrants: Persuadables (Green), Sure Things (Blue), Lost Causes (Gray), Sleeping Dogs (Red).
   * Account volume breakdown and distribution metrics.
3. **📋 CRM Target Roster:**
   * Filterable, searchable data table of all approved campaign targets.
   * Real-time metrics: Individual $\tau$, Offer Cost, Net Profit, Quadrant.
   * 1-Click **Download CRM Target Roster (.CSV)** button for Salesforce/Braze ingestion.
4. **📄 Executive Brief:**
   * Live preview of the HTML executive decision brief.
   * One-click **Print / Save as PDF** button with print stylesheet optimization.
5. **🧪 Single-Customer Simulator (What-If Analysis):**
   * Real-time feature tweaking for tenure, monthly charges, contract, tech support, etc.
   * Real-time dual-model scoring: $P(\text{Stay} \mid \text{Offer})$ vs $P(\text{Stay} \mid \text{No Offer})$.
   * Computed $\tau(X)$ and actionable prescriptive targeting recommendation.

---

# 11. Automated HTML/PDF Executive Reporting Engine (`src/report_generator.py`)

### 11.1 Architecture & Styling
* **Framework:** Pure HTML5 and Bootstrap 5 with zero external server-side dependencies.
* **Print Optimization (`@media print`):** Strips UI chrome, suppresses navigation elements, resets page margins, and enforces clean page breaks for PDF export via `Ctrl + P`.

### 11.2 Report Sections
1. **Header & Metadata:** Report title, generation timestamp, operational approval badge.
2. **Executive KPI Grid:** 4-box summary (Budget Spent, Net Profit, Customers Saved, Campaign ROI).
3. **Strategic Comparison Table:** Complete 3-way benchmark against Standard Churn and Random Promotion.
4. **Top 10 Target Accounts:** High-priority accounts sorted by expected net profit.
5. **Behavioral Cohort Breakdown:** Distribution of customer base across the 4 quadrants.

---

# 12. End-to-End Execution, Testing & Docker Deployment

### 12.1 Execution CLI Commands
* **Execute End-to-End Pipeline:**
  ```bash
  python run_pipeline.py
  ```
* **Launch Interactive Streamlit Dashboard:**
  ```bash
  streamlit run app.py
  ```
* **Run Banking & Cross-Industry Test Suite:**
  ```bash
  python test_other_dataset.py
  ```
* **Run Kaggle Customer Churn Benchmark:**
  ```bash
  python test_kaggle_shahid_azeem.py
  ```

### 12.2 Dockerfile Specification
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

# 13. Machine Learning Interview & System Defense Masterclass

### 13.1 The 30-Second Elevator Pitch
> *"Most churn projects only predict binary churn risk, which leads marketing teams to waste discounts on customers who were either going to stay anyway or were already lost causes. I built **CausalRetention**, an end-to-end prescriptive machine learning engine that uses a Two-Model (T-Learner) XGBoost architecture to estimate the Conditional Average Treatment Effect (CATE). We isolate the true 'Persuadable' accounts and feed them into a Knapsack financial optimizer. Under an identical $15,000 budget, traditional churn targeting produced a -$1,961 net loss, whereas our causal uplift engine generated +$112,762 in net preserved revenue—a 752% campaign ROI."*

---

### 13.2 Technical Deep-Dive Interview Defense

#### Q1: "Why did you use a T-Learner instead of an S-Learner or X-Learner?"
* **Answer:**  
  *"In an S-Learner, the treatment flag $T$ is just one of many features in a single model. Tree splits prioritize features with high variance like tenure or monthly charges. The treatment indicator often gets regularized or pruned, causing uplift estimates to collapse toward zero.  
  The T-Learner eliminates this treatment dilution by fitting two separate estimators: $M_1$ on treatment and $M_0$ on control.  
  While the X-Learner is preferable when treatment groups are severely unbalanced (e.g., 99% control, 1% treatment), our randomized retention pilots are balanced (~50/50). In balanced settings, the T-Learner is mathematically unbiased, avoids the estimation error of second-stage imputation models, and trains twice as fast."*

#### Q2: "How do you evaluate an Uplift Model without ground truth counterfactuals?"
* **Answer:**  
  *"Because individual counterfactuals are unobservable, standard metrics like RMSE or ROC-AUC cannot be applied. Instead, we evaluate at the cohort level using **Qini Curves and Cumulative Uplift (AUUC)** on an out-of-time holdout cohort.  
  We sort holdout customers descending by predicted uplift $\hat{\tau}(X)$ and partition them into deciles. Within each decile, we compute the incremental retention between treated and control units:  
  $$Q(k) = Y_{T, k} - \left( Y_{C, k} \times \frac{N_{T, k}}{N_{C, k}} \right)$$  
  A model that effectively isolates Persuadables will produce a steep upward curve in the early deciles, demonstrating that targeting the top 20–30% captures the vast majority of all potential incremental retentions."*

#### Q3: "Why did standard churn prediction produce a negative ROI in your benchmark?"
* **Answer:**  
  *"Traditional churn models prioritize customers with the highest churn probability. However, extreme churn probability is heavily correlated with structural dissatisfaction (e.g., severe service failure, relocation, price hikes).  
  These users are **Lost Causes**: an offer barely increases their retention probability (e.g., from 10% to 15%). When you spend $40 on discounts for these accounts, you incur the cost without preventing the churn, yielding a net loss.  
  Causal uplift models evaluate the *derivative* (the treatment effect $\Delta P$), directing budget exclusively to **Persuadables** whose retention is elastic with respect to the offer."*

#### Q4: "How does the Knapsack budget optimizer work, and why not just target everyone with positive uplift?"
* **Answer:**  
  *"A positive uplift score ($\tau > 0$) is necessary, but not sufficient for profitability. If a customer has a +5% uplift ($\tau = 0.05$) on a $20/month contract (annual CLV = $240), the expected gross revenue saved is $0.05 \times \$240 = \$12.00$. If our retention offer costs $45.00, targeting them yields a net loss of -$33.00.  
  We compute **Expected Net Profit**:  
  $$\text{NetProfit}_i = (\tau_i \times \text{CLV}_i) - \text{OfferCost}_i$$  
  Under a finite marketing budget $B$, we frame this as a 0/1 Knapsack optimization. We sort customers by expected net profit and greedily allocate budget until the marginal return equals zero or the capital cap is hit."*

#### Q5: "What prevents data leakage between your treatment and control models?"
* **Answer:**  
  *"Two safeguards are enforced:  
  1. The preprocessor (`ColumnTransformer`) is fit on the full training partition before cohort splitting.  
  2. The input feature matrix $X$ contains strictly pre-intervention covariates. Post-treatment engagement metrics (such as post-offer logins or discount redemption flags) are strictly excluded from the feature space."*

#### Q6: "How do you detect and handle Sleeping Dogs?"
* **Answer:**  
  *"Sleeping Dogs are identified as accounts where the predicted uplift is significantly negative:  
  $$\hat{\tau}(X) = M_1(X) - M_0(X) < -0.05$$  
  In subscription businesses, these are typically autopay customers with low usage. Reaching out with a retention offer serves as an unwanted reminder, prompting them to cancel. Our optimizer identifies these accounts and places them on a strict **Do Not Contact** blacklist."*
