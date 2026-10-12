import os
import sys
import pandas as pd
import numpy as np

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from dynamic_pipeline import auto_prepare_dataset, UniversalTLearner
from optimizer import apply_financial_metrics, optimize_retention_budget, compare_all_strategies
from report_generator import generate_html_report
from uplift_model import compute_cumulative_uplift_curve


def load_shahid_azeem_dataset() -> pd.DataFrame:
    """
    Loads Muhammad Shahid Azeem's Customer Churn Dataset.
    Supports both Kaggle runtime environments (/kaggle/input/...)
    and local development environments (data/raw/...).
    """
    # Possible Kaggle input paths
    kaggle_paths = [
        "/kaggle/input/datasets/muhammadshahidazeem/customer-churn-dataset/customer_churn_dataset-testing-master.csv",
        "/kaggle/input/datasets/muhammadshahidazeem/customer-churn-dataset/customer_churn_dataset-training-master.csv",
        "/kaggle/input/customer-churn-dataset/customer_churn_dataset-testing-master.csv",
        "/kaggle/input/customer-churn-dataset/customer_churn_dataset-training-master.csv",
    ]
    for p in kaggle_paths:
        if os.path.exists(p):
            print(f"Loading dataset directly from Kaggle path: {p}")
            return pd.read_csv(p)

    # Local cache paths
    local_cache = os.path.join("data", "raw", "customer_churn_dataset.csv")
    if os.path.exists(local_cache):
        print(f"Loading dataset from local cache: {local_cache}")
        return pd.read_csv(local_cache)

    # Download from mirror if not present
    url = "https://raw.githubusercontent.com/faisalghifari17/customer-churn-prediction/main/customer_churn_dataset-testing-master.csv"
    print(f"Downloading dataset from mirror: {url}")
    os.makedirs(os.path.dirname(local_cache), exist_ok=True)
    df = pd.read_csv(url)
    df.to_csv(local_cache, index=False)
    print(f"Saved locally to: {local_cache}")
    return df


def run_shahid_azeem_pipeline(sample_size: int = 15000, budget_limit: float = 15000.0):
    print("=" * 75)
    print("  KAGGLE BENCHMARK: MUHAMMAD SHAHID AZEEM CUSTOMER CHURN DATASET")
    print("=" * 75)

    df_raw = load_shahid_azeem_dataset()
    print(f"\n[Dataset Metadata]")
    print(f"-> Total Rows: {len(df_raw):,}")
    print(f"-> Total Columns: {len(df_raw.columns)}")
    print(f"-> Columns: {df_raw.columns.tolist()}")
    print(f"-> Churn Rate: {df_raw['Churn'].mean() * 100:.1f}%")
    print(f"-> Total Spend (Mean): ${df_raw['Total Spend'].mean():,.2f} | (Median): ${df_raw['Total Spend'].median():,.2f}")

    # Step 1: Auto Prepare Dataset
    print("\n[Step 1/5] Auto-Preparing Dataset for Causal Retention...")
    df_prep, feat_cols = auto_prepare_dataset(
        df=df_raw,
        target_col="Churn",
        target_is_churn=True,
    )

    print(f"-> Customer ID Column: customerID (e.g. {df_prep['customerID'].iloc[0]})")
    print(f"-> Features Selected ({len(feat_cols)}): {feat_cols}")
    print(f"-> Annual CLV derived from Total Spend: Mean = ${df_prep['annual_clv'].mean():,.2f}")
    print(f"-> Offer Cost (8% of CLV): Mean = ${df_prep['offer_cost'].mean():,.2f}")

    # Subsample for fast training if large
    if sample_size and len(df_prep) > sample_size:
        print(f"-> Subsampling {sample_size:,} accounts for high-speed model fitting & evaluation...")
        df_eval = df_prep.sample(sample_size, random_state=42).reset_index(drop=True)
    else:
        df_eval = df_prep

    # Step 2: Model Training
    print("\n[Step 2/5] Training Two-Model XGBoost Universal T-Learner...")
    engine = UniversalTLearner(random_state=42)
    engine.fit(df_eval, feat_cols)
    print("-> Universal T-Learner trained successfully!")

    # Step 3: Model Scoring & Cumulative Uplift Curve
    print("\n[Step 3/5] Scoring Uplift & Calculating Cumulative Curve...")
    scored = engine.predict_uplift(df_eval)
    curve = compute_cumulative_uplift_curve(scored, score_col="uplift_score")
    print("\nTop Deciles of Cumulative Uplift (Qini Curve):")
    print(curve.head(5).to_string(index=False))

    # Step 4: Financial Knapsack Optimization
    print(f"\n[Step 4/5] Running Financial Optimization (${budget_limit:,.2f} Budget)...")
    scored_fin = apply_financial_metrics(scored)
    opt_result = optimize_retention_budget(scored_fin, budget_limit=budget_limit)
    summary = opt_result["summary"]
    comp_df = compare_all_strategies(scored_fin, budget_limit=budget_limit)

    print(f"\nOptimization Summary for ${budget_limit:,.2f} Budget:")
    print(f"-> Budget Spent: ${summary['budget_spent']:,.2f} (Remaining: ${summary['budget_remaining']:,.2f})")
    print(f"-> Customers Targeted: {summary['customers_targeted']}")
    print(f"-> Incremental Retained Lift: +{summary['expected_retained_lift']:.1f} customers")
    print(f"-> Gross Revenue Saved: ${summary['gross_revenue_saved']:,.2f}")
    print(f"-> Net Profit Generated: ${summary['net_profit']:,.2f}")
    print(f"-> Campaign Net ROI: +{summary['roi_percent']:.1f}%")
    print(f"-> Quadrant Breakdown: {summary['quadrant_breakdown']}")

    print("\nBenchmark Strategy Comparison:")
    print(comp_df.to_string(index=False))

    # Step 5: Report Generation
    print("\n[Step 5/5] Generating Executive Decision Brief...")
    report_path = "reports/shahid_azeem_churn_report.html"
    generate_html_report(
        summary=summary,
        comparison_df=comp_df,
        targeted_df=opt_result["targeted_df"],
        output_path=report_path,
    )
    print(f"-> Report saved to: {report_path}")

    # Save target CSV
    roster_path = "reports/shahid_azeem_target_roster.csv"
    disp_cols = ["customerID", "uplift_score", "campaign_offer_cost", "expected_net_profit", "quadrant"]
    for c in ["Subscription Type", "Contract Length", "Total Spend", "Usage Frequency", "Support Calls"]:
        if c in opt_result["targeted_df"].columns:
            disp_cols.insert(1, c)
    opt_result["targeted_df"][disp_cols].to_csv(roster_path, index=False)
    print(f"-> Target dispatch CSV saved to: {roster_path}")

    print("\n" + "=" * 75)
    print(" PIPELINE COMPLETED SUCCESSFULLY ON SHAHID AZEEM DATASET!")
    print("=" * 75)
    return summary, comp_df


if __name__ == "__main__":
    run_shahid_azeem_pipeline()
