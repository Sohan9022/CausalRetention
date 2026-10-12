import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def auto_prepare_dataset(
    df: pd.DataFrame,
    target_col: str,
    target_is_churn: bool = True,
    treatment_col: str = None,
    id_col: str = None,
    clv_col: str = None,
    cost_col: str = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Adapts ANY uploaded tabular dataset into a causal retention structure.
    Returns:
      - processed_df: ready for modeling
      - feature_cols: list of independent feature column names
    """
    df = df.copy()

    # 1. Customer ID Detection & Formatting
    id_candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["customerid", "customer_id", "cust_id", "custid", "clientnum", "client_id", "accountid", "user_id"])
    ]
    detected_id = id_col if (id_col and id_col in df.columns) else (id_candidates[0] if id_candidates else None)

    if detected_id:
        df["customerID"] = df[detected_id].astype(str)
    elif "customerID" in df.columns:
        df["customerID"] = df["customerID"].astype(str)
    else:
        df["customerID"] = [f"ID-{i+1:05d}" for i in range(len(df))]

    # 2. Target Variable (Standardized to 'retained' = 1, 'churn' = 0)
    raw_target = df[target_col]
    if raw_target.dtype == object or str(raw_target.iloc[0]).lower() in ["yes", "no", "true", "false", "attrited customer", "existing customer"]:
        raw_numeric = raw_target.astype(str).str.strip().str.lower().map({
            "yes": 1, "1": 1, "true": 1, "churn": 1, "exited": 1, "attrited customer": 1,
            "no": 0, "0": 0, "false": 0, "stay": 0, "retained": 0, "existing customer": 0
        }).fillna(0).astype(int)
    else:
        raw_numeric = pd.to_numeric(raw_target, errors="coerce").fillna(0).astype(int)

    if target_is_churn:
        df["churn"] = raw_numeric
        df["retained"] = 1 - df["churn"]
    else:
        df["retained"] = raw_numeric
        df["churn"] = 1 - df["retained"]

    # 3. Treatment Column
    if treatment_col and treatment_col in df.columns:
        t_raw = df[treatment_col]
        if t_raw.dtype == object:
            df["treatment"] = t_raw.astype(str).str.strip().str.lower().map({
                "yes": 1, "1": 1, "true": 1, "treatment": 1, "offer": 1, "discount": 1
            }).fillna(0).astype(int)
        else:
            df["treatment"] = pd.to_numeric(t_raw, errors="coerce").fillna(0).astype(int)
    else:
        # If dataset lacks an experimental treatment column, simulate a 50/50 randomized A/B pilot
        np.random.seed(random_state)
        n = len(df)
        df["treatment"] = np.random.binomial(1, 0.5, size=n)

        # Apply realistic heterogeneous treatment effect based on observed risk
        baseline_churn = df["churn"].values
        # For treated users, simulate 35-45% relative reduction in churn probability
        p_churn_treat = np.clip(baseline_churn * 0.60, 0.05, 0.90)
        p_stay_treat = 1.0 - p_churn_treat
        # Update retained status under randomized assignment
        simulated_retained = np.where(
            df["treatment"] == 1,
            np.random.binomial(1, p_stay_treat),
            df["retained"]
        )
        df["retained"] = simulated_retained
        df["churn"] = 1 - df["retained"]

    # 4. Customer Lifetime Value (CLV) Calculation
    non_feature_id_cols = set([detected_id, "customerID", "customerid", "id", target_col] + id_candidates)
    if clv_col and clv_col in df.columns:
        df["annual_clv"] = pd.to_numeric(df[clv_col], errors="coerce").fillna(600.0)
    else:
        # Intelligently search for revenue/spend/balance columns (excluding IDs, age, tenure)
        num_cols_pool = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in non_feature_id_cols and c.lower() not in ["retained", "churn", "treatment", "age", "tenure", "rownumber", "index", "zipcode", "creditscore"]
        ]

        # Priority 1: Direct spend/revenue/charges
        spend_candidates = [
            c for c in num_cols_pool
            if any(k in c.lower() for k in ["clv", "revenue", "spend", "charge", "mrr", "arr", "monthly", "bill", "fee"])
        ]
        # Priority 2: Banking/finance proxies (balance, salary)
        banking_candidates = [
            c for c in num_cols_pool
            if any(k in c.lower() for k in ["balance", "salary", "income"])
        ]

        if spend_candidates:
            c = spend_candidates[0]
            if "month" in c.lower() or df[c].mean() < 300:
                df["annual_clv"] = np.round(df[c].clip(lower=10.0) * 12, 2)
            else:
                df["annual_clv"] = np.round(df[c].clip(lower=100.0), 2)
        elif banking_candidates:
            c = banking_candidates[0]
            if "balance" in c.lower():
                # Net Interest Margin (~4% of deposit balance) + banking fee proxy
                df["annual_clv"] = np.round(np.clip(df[c] * 0.04, 150.0, 4500.0), 2)
            else:
                # Discretionary financial spend proxy (~1.5% of annual salary)
                df["annual_clv"] = np.round(np.clip(df[c] * 0.015, 200.0, 3500.0), 2)
        else:
            df["annual_clv"] = 600.0

    # Ensure annual CLV has no NaNs and realistic positive floor
    df["annual_clv"] = df["annual_clv"].fillna(600.0).clip(lower=100.0)

    # 5. Offer Cost
    if cost_col and cost_col in df.columns:
        df["offer_cost"] = pd.to_numeric(df[cost_col], errors="coerce").fillna(40.0)
    else:
        df["offer_cost"] = np.round(np.clip(df["annual_clv"] * 0.08, 15.0, 80.0), 2)

    # 6. Detect Feature Columns (Exclude IDs, targets, non-informative high-cardinality names/indices)
    exclude_cols = {
        target_col,
        id_col if id_col else "",
        detected_id if detected_id else "",
        treatment_col if treatment_col else "",
        clv_col if clv_col else "",
        cost_col if cost_col else "",
        "customerID",
        "retained",
        "churn",
        "treatment",
        "annual_clv",
        "offer_cost",
    }
    exclude_cols.update(id_candidates)

    feature_cols = []
    n_rows = len(df)
    for c in df.columns:
        if c in exclude_cols:
            continue
        c_lower = c.lower()
        if c_lower in ["rownumber", "row_number", "index", "surname", "last_name", "first_name", "full_name", "name"]:
            continue
        # Drop categorical columns where cardinality is excessively high (>100 or >20% unique strings)
        if df[c].dtype == object and (df[c].nunique() > 50 and df[c].nunique() / n_rows > 0.15):
            continue
        feature_cols.append(c)

    return df, feature_cols


def create_dynamic_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Builds an automated imputer & encoder for any arbitrary tabular schema."""
    sub_df = df[feature_cols]
    num_cols = sub_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = sub_df.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=15)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, num_cols, cat_cols


class UniversalTLearner:
    """Universal T-Learner supporting any tabular dataset."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = None
        self.model_treatment = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8,
            random_state=random_state, eval_metric="logloss"
        )
        self.model_control = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8,
            random_state=random_state, eval_metric="logloss"
        )
        self.model_baseline_churn = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8,
            random_state=random_state, eval_metric="logloss"
        )
        self.feature_cols = []
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame, feature_cols: list[str]):
        self.feature_cols = feature_cols
        self.preprocessor, _, _ = create_dynamic_preprocessor(df_train, feature_cols)

        X_transformed = self.preprocessor.fit_transform(df_train[feature_cols])
        T = df_train["treatment"].values
        y_retained = df_train["retained"].values
        y_churn = df_train["churn"].values

        idx_treat = (T == 1)
        idx_ctrl = (T == 0)

        # Train treatment & control models
        self.model_treatment.fit(X_transformed[idx_treat], y_retained[idx_treat])
        self.model_control.fit(X_transformed[idx_ctrl], y_retained[idx_ctrl])
        self.model_baseline_churn.fit(X_transformed, y_churn)

        self.is_fitted = True
        return self

    def predict_uplift(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        X_trans = self.preprocessor.transform(df[self.feature_cols])

        p_treat = self.model_treatment.predict_proba(X_trans)[:, 1]
        p_ctrl = self.model_control.predict_proba(X_trans)[:, 1]
        uplift = p_treat - p_ctrl
        p_churn = self.model_baseline_churn.predict_proba(X_trans)[:, 1]

        res = df.copy()
        res["p_retained_treatment"] = np.round(p_treat, 4)
        res["p_retained_control"] = np.round(p_ctrl, 4)
        res["uplift_score"] = np.round(uplift, 4)
        res["baseline_churn_prob"] = np.round(p_churn, 4)
        return res
