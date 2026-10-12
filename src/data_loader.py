import os
import urllib.request
import numpy as np
import pandas as pd

RAW_DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_FILE_PATH = os.path.join("data", "raw", "Telco-Customer-Churn.csv")
PROCESSED_FILE_PATH = os.path.join("data", "processed", "telco_retention_experiment.csv")


def download_raw_telco_data(dest_path: str = RAW_FILE_PATH) -> pd.DataFrame:
    """Download the official IBM Telco Customer Churn dataset if not present."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        print(f"Downloading raw Telco dataset from {RAW_DATA_URL}...")
        urllib.request.urlretrieve(RAW_DATA_URL, dest_path)
        print(f"Saved to {dest_path}")
    df = pd.read_csv(dest_path)
    return df


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data types, handle missing values in TotalCharges, and format categorical fields."""
    df = df.copy()

    # TotalCharges contains empty strings for new users (tenure == 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Convert SeniorCitizen to category string for consistent preprocessing
    df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})

    return df


def generate_retention_experiment(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Transforms the Telco dataset into a causal retention experiment.
    Simulates a 50/50 randomized A/B trial (Treatment vs. Control) with ground-truth
    behavioral response cohorts:
      - Persuadables: Price/contract sensitive, stay IF given retention offer.
      - Sure Things: High tenure & long contract, stay regardless of offer.
      - Lost Causes: Extreme churn indicators, leave even if given offer.
      - Sleeping Dogs: Low engagement, contacting them causes slight cancellation spike.
    """
    np.random.seed(random_state)
    df = clean_telco_data(df)
    n = len(df)

    # 1. Randomized A/B Assignment (50% treatment, 50% control)
    treatment = np.random.binomial(1, 0.5, size=n)
    df["treatment"] = treatment

    # 2. Financial Context: Customer Lifetime Value (CLV) & Offer Cost
    # Expected Annual Value = MonthlyCharges * 12
    # Contract discount cost: flat $15 administrative cost + 15% discount for 3 months
    df["annual_clv"] = np.round(df["MonthlyCharges"] * 12, 2)
    df["offer_cost"] = np.round(15.0 + (df["MonthlyCharges"] * 0.15 * 3), 2)

    # 3. Latent Behavioral Archetype Modeling
    # Baseline churn tendency based on observable domain signals
    is_m2m = (df["Contract"] == "Month-to-month").astype(float)
    is_fiber = (df["InternetService"] == "Fiber optic").astype(float)
    has_no_support = (df["TechSupport"] == "No").astype(float)
    tenure_norm = 1.0 - (df["tenure"] / 72.0).clip(0, 1)  # Low tenure = higher risk
    high_charges = (df["MonthlyCharges"] > 70).astype(float)
    electronic_check = (df["PaymentMethod"] == "Electronic check").astype(float)

    # Composite baseline churn risk score [0, 1]
    baseline_score = (
        0.35 * is_m2m +
        0.20 * tenure_norm +
        0.15 * high_charges +
        0.15 * has_no_support +
        0.10 * is_fiber +
        0.05 * electronic_check
    )

    # Latent probability of retention without offer (Control) and with offer (Treatment)
    p_retained_control = np.clip(1.0 - baseline_score + np.random.normal(0, 0.05, n), 0.02, 0.98)
    p_retained_treatment = p_retained_control.copy()

    # Define uplift archetypes
    for i in range(n):
        # A. Sure Things: Loyal users (2-year contract, tenure > 36)
        if df.iloc[i]["Contract"] == "Two year" or df.iloc[i]["tenure"] > 40:
            p_retained_control[i] = np.random.uniform(0.90, 0.99)
            p_retained_treatment[i] = np.random.uniform(0.92, 0.99)

        # B. Lost Causes: High charges, month-to-month, low tenure, no support
        elif baseline_score.iloc[i] > 0.75:
            p_retained_control[i] = np.random.uniform(0.05, 0.20)
            p_retained_treatment[i] = np.random.uniform(0.10, 0.25)  # Offer barely helps

        # C. Sleeping Dogs: Auto-pay, paperless, low charges, haven't touched support
        elif (df.iloc[i]["PaperlessBilling"] == "Yes" and
              df.iloc[i]["MonthlyCharges"] < 35 and
              df.iloc[i]["TechSupport"] == "No internet service"):
            p_retained_control[i] = 0.92
            p_retained_treatment[i] = 0.78  # Contacting them triggers cancellation!

        # D. Persuadables: Moderate baseline risk, month-to-month or 1-year, price-conscious
        else:
            p_retained_control[i] = np.random.uniform(0.30, 0.55)
            # High treatment uplift!
            p_retained_treatment[i] = np.clip(p_retained_control[i] + np.random.uniform(0.30, 0.45), 0.0, 0.95)

    # Assign observed retention based on whether customer actually received treatment
    p_observed = np.where(treatment == 1, p_retained_treatment, p_retained_control)
    observed_retained = np.random.binomial(1, p_observed)

    # Add outcome variables
    df["retained"] = observed_retained
    df["churn"] = 1 - observed_retained

    # Ground truth uplift for evaluation benchmarks (unobserved in real life, used for validation)
    df["true_uplift"] = np.round(p_retained_treatment - p_retained_control, 4)

    return df


def load_or_create_data(force_refresh: bool = False) -> pd.DataFrame:
    """Loads processed retention experiment dataset, creating it if needed."""
    if os.path.exists(PROCESSED_FILE_PATH) and not force_refresh:
        return pd.read_csv(PROCESSED_FILE_PATH)

    raw_df = download_raw_telco_data()
    processed_df = generate_retention_experiment(raw_df)
    os.makedirs(os.path.dirname(PROCESSED_FILE_PATH), exist_ok=True)
    processed_df.to_csv(PROCESSED_FILE_PATH, index=False)
    print(f"Processed retention experiment saved to {PROCESSED_FILE_PATH}")
    return processed_df


if __name__ == "__main__":
    df = load_or_create_data(force_refresh=True)
    print("\nDataset Summary:")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nTreatment Distribution:")
    print(df["treatment"].value_counts(normalize=True))
    print("\nRetention Rate by Group:")
    print(df.groupby("treatment")["retained"].mean())
    print("\nMean Annual CLV:", df["annual_clv"].mean())
    print("Mean Offer Cost:", df["offer_cost"].mean())
