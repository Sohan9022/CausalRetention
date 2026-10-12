import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from feature_pipeline import ALL_FEATURE_COLS, create_preprocessor, get_feature_names

MODELS_DIR = "models"


class TLearnerUpliftEngine:
    """
    Two-Model (T-Learner) Causal Uplift Engine using Scikit-Learn & XGBoost.

    T-Learner trains two separate estimators:
      - Model Treatment (M_1): Predicts P(Retained | X, Treatment=1)
      - Model Control (M_0): Predicts P(Retained | X, Treatment=0)

    Predicted Uplift (Individual Treatment Effect, tau):
      tau(X) = P_hat(Retained | X, T=1) - P_hat(Retained | X, T=0)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = create_preprocessor()
        self.model_treatment = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss",
        )
        self.model_control = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss",
        )
        # Standard Churn classifier for benchmark comparison
        self.model_baseline_churn = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss",
        )
        self.is_fitted = False
        self.feature_names = None

    def fit(self, df_train: pd.DataFrame):
        """Fit preprocessor, treatment model, control model, and baseline model."""
        X_raw = df_train[ALL_FEATURE_COLS]
        T = df_train["treatment"].values
        y_retained = df_train["retained"].values
        y_churn = df_train["churn"].values

        # 1. Fit preprocessor and transform features
        X_transformed = self.preprocessor.fit_transform(X_raw)
        self.feature_names = get_feature_names(self.preprocessor)

        # 2. Split into Treatment and Control subsets
        idx_treat = (T == 1)
        idx_ctrl = (T == 0)

        X_treat, y_treat = X_transformed[idx_treat], y_retained[idx_treat]
        X_ctrl, y_ctrl = X_transformed[idx_ctrl], y_retained[idx_ctrl]

        # 3. Train Treatment & Control models
        self.model_treatment.fit(X_treat, y_treat)
        self.model_control.fit(X_ctrl, y_ctrl)

        # 4. Train standard baseline churn model (predicts raw churn regardless of treatment)
        self.model_baseline_churn.fit(X_transformed, y_churn)

        self.is_fitted = True
        return self

    def predict_uplift(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates:
          - p_retained_treatment: P(Stay | Offer)
          - p_retained_control: P(Stay | No Offer)
          - uplift_score (tau): p_retained_treatment - p_retained_control
          - baseline_churn_prob: standard churn probability
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        X_transformed = self.preprocessor.transform(df[ALL_FEATURE_COLS])

        # Predict probability of retention (class 1)
        p_treat = self.model_treatment.predict_proba(X_transformed)[:, 1]
        p_ctrl = self.model_control.predict_proba(X_transformed)[:, 1]
        uplift = p_treat - p_ctrl

        # Standard baseline churn probability
        p_churn = self.model_baseline_churn.predict_proba(X_transformed)[:, 1]

        results = df.copy()
        results["p_retained_treatment"] = np.round(p_treat, 4)
        results["p_retained_control"] = np.round(p_ctrl, 4)
        results["uplift_score"] = np.round(uplift, 4)
        results["baseline_churn_prob"] = np.round(p_churn, 4)

        return results

    def save(self, directory: str = MODELS_DIR):
        """Save fitted model bundle."""
        os.makedirs(directory, exist_ok=True)
        bundle = {
            "preprocessor": self.preprocessor,
            "model_treatment": self.model_treatment,
            "model_control": self.model_control,
            "model_baseline_churn": self.model_baseline_churn,
            "feature_names": self.feature_names,
            "random_state": self.random_state,
        }
        path = os.path.join(directory, "uplift_model_bundle.joblib")
        joblib.dump(bundle, path)
        print(f"Model bundle saved to {path}")

    @classmethod
    def load(cls, directory: str = MODELS_DIR) -> "TLearnerUpliftEngine":
        """Load fitted model bundle."""
        path = os.path.join(directory, "uplift_model_bundle.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model bundle not found at {path}")
        bundle = joblib.load(path)
        engine = cls(random_state=bundle.get("random_state", 42))
        engine.preprocessor = bundle["preprocessor"]
        engine.model_treatment = bundle["model_treatment"]
        engine.model_control = bundle["model_control"]
        engine.model_baseline_churn = bundle["model_baseline_churn"]
        engine.feature_names = bundle["feature_names"]
        engine.is_fitted = True
        return engine


def compute_cumulative_uplift_curve(df_scored: pd.DataFrame, score_col: str, n_bins: int = 10) -> pd.DataFrame:
    """
    Computes cumulative gains / Qini curve metrics for a given scoring column.
    """
    df_sorted = df_scored.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    df_sorted["bin"] = pd.qcut(df_sorted.index, q=n_bins, labels=False)

    cum_data = []
    total_customers = len(df_sorted)

    for b in range(n_bins):
        subset = df_sorted[df_sorted["bin"] <= b]
        n_sub = len(subset)

        # In treatment group
        treat_sub = subset[subset["treatment"] == 1]
        n_t = len(treat_sub)
        y_t = treat_sub["retained"].sum()

        # In control group
        ctrl_sub = subset[subset["treatment"] == 0]
        n_c = len(ctrl_sub)
        y_c = ctrl_sub["retained"].sum()

        # Scaled uplift = y_t - (y_c * (n_t / n_c)) if n_c > 0 else 0
        incremental_retained = y_t - (y_c * (n_t / n_c)) if n_c > 0 else 0

        cum_data.append({
            "decile": b + 1,
            "targeted_customers": n_sub,
            "percent_targeted": round((n_sub / total_customers) * 100, 1),
            "treatment_retained": y_t,
            "control_retained": y_c,
            "cumulative_incremental_retained": round(incremental_retained, 1),
        })

    return pd.DataFrame(cum_data)


if __name__ == "__main__":
    from data_loader import load_or_create_data

    df = load_or_create_data()
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

    print(f"Training T-Learner on {len(train_df)} rows, testing on {len(test_df)} rows...")
    engine = TLearnerUpliftEngine(random_state=42)
    engine.fit(train_df)
    engine.save()

    scored_test = engine.predict_uplift(test_df)
    print("\nSample Scored Test Predictions:")
    print(scored_test[["customerID", "p_retained_control", "p_retained_treatment", "uplift_score", "baseline_churn_prob"]].head())

    curve = compute_cumulative_uplift_curve(scored_test, score_col="uplift_score")
    print("\nCumulative Uplift Curve (Deciles):")
    print(curve)
