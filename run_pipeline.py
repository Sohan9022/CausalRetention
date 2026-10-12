"""
End-to-end pipeline runner for CausalRetention.
Runs data loading, model training, evaluation, financial optimization, and report generation.
"""

import os
import sys

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_or_create_data
from feature_pipeline import ALL_FEATURE_COLS
from optimizer import apply_financial_metrics, compare_all_strategies, optimize_retention_budget
from report_generator import generate_html_report
from uplift_model import TLearnerUpliftEngine, compute_cumulative_uplift_curve
from sklearn.model_selection import train_test_split


def main():
    print("=" * 70)
    print(" CAUSALRETENTION: UPLIFT MODELING & RETENTION DECISION ENGINE")
    print("=" * 70)

    # 1. Data Preparation
    print("\n[Step 1/5] Loading and formatting retention experiment data...")
    df = load_or_create_data(force_refresh=False)
    print(f"-> Total accounts loaded: {len(df):,}")

    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
    print(f"-> Train size: {len(train_df):,} | Test size: {len(test_df):,}")

    # 2. Model Training
    print("\n[Step 2/5] Training T-Learner XGBoost Uplift Models (Treatment & Control)...")
    engine = TLearnerUpliftEngine(random_state=42)
    engine.fit(train_df)
    engine.save()
    print("-> Model training complete & saved to models/uplift_model_bundle.joblib")

    # 3. Model Scoring & Evaluation
    print("\n[Step 3/5] Scoring test cohort & calculating uplift curve...")
    scored_test = engine.predict_uplift(test_df)
    curve = compute_cumulative_uplift_curve(scored_test, score_col="uplift_score")
    print("\nTop 5 Deciles of Cumulative Uplift:")
    print(curve.head().to_string(index=False))

    # 4. Financial Optimization
    print("\n[Step 4/5] Running Knapsack Retention Budget Optimizer...")
    scored_full = engine.predict_uplift(df)
    scored_fin = apply_financial_metrics(scored_full)

    budget_limit = 15000.0
    opt_result = optimize_retention_budget(scored_fin, budget_limit=budget_limit)
    summary = opt_result["summary"]
    comp_df = compare_all_strategies(scored_fin, budget_limit=budget_limit)

    print(f"\nOptimization Summary for ${budget_limit:,.2f} Budget:")
    print(f"-> Customers Targeted: {summary['customers_targeted']}")
    print(f"-> Customers Saved: +{summary['expected_retained_lift']:.0f}")
    print(f"-> Net Profit Generated: ${summary['net_profit']:,.2f}")
    print(f"-> Campaign ROI: +{summary['roi_percent']:.1f}%")

    print("\nBenchmark Strategy Comparison:")
    print(comp_df.to_string(index=False))

    # 5. Executive Report Generation
    print("\n[Step 5/5] Generating Executive Decision Brief...")
    report_path = generate_html_report(
        summary=summary,
        comparison_df=comp_df,
        targeted_df=opt_result["targeted_df"],
        output_path="reports/executive_retention_report.html",
    )
    print(f"-> Board-ready HTML report available at: {report_path}")

    # Save target dispatch CSV for CRM
    dispatch_path = "reports/target_campaign_roster.csv"
    opt_result["targeted_df"][
        [
            "customerID",
            "Contract",
            "tenure",
            "MonthlyCharges",
            "uplift_score",
            "campaign_offer_cost",
            "expected_net_profit",
            "quadrant",
        ]
    ].to_csv(dispatch_path, index=False)
    print(f"-> Actionable CRM target list saved to: {dispatch_path}")

    print("\n" + "=" * 70)
    print(" Pipeline execution finished successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
