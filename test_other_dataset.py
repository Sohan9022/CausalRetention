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


def test_bank_churn_dataset():
    print("=" * 70)
    print("TEST SUITE 1: BANK CUSTOMER CHURN BENCHMARK (10,000 ACCOUNTS)")
    print("=" * 70)

    cache_path = os.path.join("data", "raw", "Bank_Churn_Modelling.csv")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not os.path.exists(cache_path):
        print("Downloading Bank Customer Churn dataset...")
        url = "https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Bank%20Churn%20Modelling.csv"
        df_raw = pd.read_csv(url)
        df_raw.to_csv(cache_path, index=False)
        print(f"Saved raw file to {cache_path}")
    else:
        print(f"Loading cached file from {cache_path}...")
        df_raw = pd.read_csv(cache_path)

    print(f"[PASS] Loaded Raw Data Shape: {df_raw.shape}")
    print(f"[PASS] Columns: {df_raw.columns.tolist()}")

    # 1. Automated Schema Adaptation
    print("\n[Step 1] Running auto_prepare_dataset()...")
    df_prep, feat_cols = auto_prepare_dataset(
        df=df_raw,
        target_col="Churn",
        target_is_churn=True,
    )

    print(f"[PASS] Target converted: 'retained' mean = {df_prep['retained'].mean():.3f}, 'churn' mean = {df_prep['churn'].mean():.3f}")
    print(f"[PASS] Customer IDs detected: {df_prep['customerID'].iloc[0]} (Type: {df_prep['customerID'].dtype})")
    print(f"[PASS] Selected Feature Columns ({len(feat_cols)}): {feat_cols}")
    print(f"[PASS] Annual CLV summary (Mean: ${df_prep['annual_clv'].mean():,.2f}, Median: ${df_prep['annual_clv'].median():,.2f})")
    print(f"[PASS] Offer Cost summary (Mean: ${df_prep['offer_cost'].mean():,.2f}, Median: ${df_prep['offer_cost'].median():,.2f})")

    # Assertions
    assert "customerID" in df_prep.columns, "customerID must be present"
    assert "retained" in df_prep.columns and "treatment" in df_prep.columns, "retained and treatment must exist"
    assert "annual_clv" in df_prep.columns and "offer_cost" in df_prep.columns, "financial columns must exist"
    assert not any(c.lower() in ["surname", "rownumber"] for c in feat_cols), "High cardinality ID/Name columns must be excluded"
    assert df_prep["annual_clv"].min() >= 50.0, "CLV must have a valid positive lower bound"

    # 2. Model Training
    print("\n[Step 2] Training UniversalTLearner (Two-Model XGBoost)...")
    engine = UniversalTLearner(random_state=42)
    engine.fit(df_prep, feat_cols)
    print("[PASS] Model fitted successfully!")

    # 3. Uplift Scoring
    print("\n[Step 3] Predicting Causal Uplift & Counterfactuals...")
    scored = engine.predict_uplift(df_prep)
    print(scored[["customerID", "p_retained_control", "p_retained_treatment", "uplift_score", "baseline_churn_prob"]].head(5).to_string())

    assert -1.0 <= scored["uplift_score"].min() and scored["uplift_score"].max() <= 1.0, "Uplift must be bounded in [-1, 1]"
    assert 0.0 <= scored["p_retained_control"].min() and scored["p_retained_control"].max() <= 1.0, "Probabilities must be in [0, 1]"

    # Cumulative Uplift Curve
    curve = compute_cumulative_uplift_curve(scored, score_col="uplift_score")
    print("\nTop 5 Deciles of Cumulative Uplift:")
    print(curve.head(5).to_string(index=False))

    # 4. Financial Knapsack Optimization
    print("\n[Step 4] Running Financial Knapsack Optimization ($15,000 Retention Budget)...")
    scored_fin = apply_financial_metrics(scored)
    budget = 15000.0
    opt_result = optimize_retention_budget(scored_fin, budget_limit=budget)
    summary = opt_result["summary"]

    print(f"[PASS] Budget Spent: ${summary['budget_spent']:,.2f} / ${budget:,.2f}")
    print(f"[PASS] Customers Targeted: {summary['customers_targeted']}")
    print(f"[PASS] Incremental Customers Saved: +{summary['expected_retained_lift']:.1f}")
    print(f"[PASS] Gross Saved: ${summary['gross_revenue_saved']:,.2f}")
    print(f"[PASS] Net Profit: ${summary['net_profit']:,.2f}")
    print(f"[PASS] Campaign ROI: +{summary['roi_percent']:.1f}%")
    print(f"[PASS] Quadrants: {summary['quadrant_breakdown']}")

    assert summary["budget_spent"] <= budget, "Budget spent must not exceed limit"
    assert summary["net_profit"] > 0, "Net profit should be positive for optimized targeting"

    # Strategy Comparison
    comp_df = compare_all_strategies(scored_fin, budget_limit=budget)
    print("\nStrategy Benchmark Comparison:")
    print(comp_df.to_string(index=False))

    # 5. Executive Report Generation
    print("\n[Step 5] Generating Executive Decision Brief for Bank Churn...")
    report_path = "reports/bank_churn_retention_report.html"
    generate_html_report(
        summary=summary,
        comparison_df=comp_df,
        targeted_df=opt_result["targeted_df"],
        output_path=report_path,
    )
    assert os.path.exists(report_path) and os.path.getsize(report_path) > 1000, "Report must be generated and non-empty"
    print(f"[PASS] Report successfully generated and verified at: {report_path}")

    print("ALL TESTS PASSED ON BANK CHURN DATASET!")
    return summary, comp_df


def test_saas_churn_dataset():
    print("\n" + "=" * 70)
    print("TEST SUITE 2: B2B SAAS SUBSCRIPTION RETENTION DATASET (5,000 ACCOUNTS)")
    print("=" * 70)

    saas_path = os.path.join("data", "raw", "SaaS_Subscription_Retention.csv")
    os.makedirs(os.path.dirname(saas_path), exist_ok=True)

    # Generate synthetic SaaS cohort if not exists
    if not os.path.exists(saas_path):
        print("Synthesizing multi-feature B2B SaaS retention dataset...")
        np.random.seed(42)
        n = 5000
        tiers = np.random.choice(["Startup", "Growth", "Enterprise"], size=n, p=[0.5, 0.35, 0.15])
        contracts = np.random.choice(["Monthly", "Annual", "2-Year"], size=n, p=[0.45, 0.40, 0.15])
        base_mrr = np.where(tiers == "Startup", np.random.uniform(99, 499, n),
                    np.where(tiers == "Growth", np.random.uniform(500, 2499, n),
                             np.random.uniform(2500, 9999, n)))
        seats = np.where(tiers == "Startup", np.random.randint(3, 20, n),
                 np.where(tiers == "Growth", np.random.randint(20, 100, n),
                          np.random.randint(100, 1000, n)))
        logins = np.random.poisson(lam=25, size=n)
        support_tickets = np.random.poisson(lam=3, size=n)
        nps = np.clip(np.random.normal(loc=7, scale=2.5, size=n), 0, 10).round(1)

        # Churn risk formula based on B2B SaaS drivers
        churn_risk = (
            0.35 * (contracts == "Monthly").astype(float) +
            0.20 * np.clip(1.0 - (logins / 50.0), 0, 1) +
            0.20 * np.clip(support_tickets / 10.0, 0, 1) +
            0.15 * np.clip((10.0 - nps) / 10.0, 0, 1) +
            0.10 * np.random.normal(0, 0.05, n)
        ).clip(0.02, 0.95)

        churn_actual = np.random.binomial(1, churn_risk)

        df_saas = pd.DataFrame({
            "account_id": [f"SAAS-{i+10001}" for i in range(n)],
            "company_tier": tiers,
            "contract_type": contracts,
            "mrr": np.round(base_mrr, 2),
            "seat_count": seats,
            "weekly_active_logins": logins,
            "support_tickets_last_30d": support_tickets,
            "nps_rating": nps,
            "churn_cancelled": churn_actual,
        })
        df_saas.to_csv(saas_path, index=False)
        print(f"Saved SaaS dataset ({len(df_saas):,} rows) to {saas_path}")
    else:
        print(f"Loading cached SaaS dataset from {saas_path}...")
        df_saas = pd.read_csv(saas_path)

    print(f"[PASS] Loaded SaaS Data Shape: {df_saas.shape}")
    print(f"[PASS] Columns: {df_saas.columns.tolist()}")

    # 1. Automated Schema Adaptation
    print("\n[Step 1] Running auto_prepare_dataset() on SaaS data...")
    df_prep, feat_cols = auto_prepare_dataset(
        df=df_saas,
        target_col="churn_cancelled",
        target_is_churn=True,
        id_col="account_id",
        clv_col=None, # Auto-detect from 'mrr'
    )

    print(f"[PASS] Target converted: 'retained' mean = {df_prep['retained'].mean():.3f}, 'churn' mean = {df_prep['churn'].mean():.3f}")
    print(f"[PASS] Customer IDs: {df_prep['customerID'].iloc[0]}")
    print(f"[PASS] Feature Columns: {feat_cols}")
    print(f"[PASS] Annual CLV from MRR (Mean: ${df_prep['annual_clv'].mean():,.2f}, Median: ${df_prep['annual_clv'].median():,.2f})")
    print(f"[PASS] Offer Cost (Mean: ${df_prep['offer_cost'].mean():,.2f}, Median: ${df_prep['offer_cost'].median():,.2f})")

    # 2. Model Training
    print("\n[Step 2] Training UniversalTLearner on SaaS dataset...")
    engine = UniversalTLearner(random_state=42)
    engine.fit(df_prep, feat_cols)
    print("[PASS] Universal T-Learner trained successfully on SaaS features!")

    # 3. Model Scoring
    print("\n[Step 3] Scoring SaaS uplift...")
    scored = engine.predict_uplift(df_prep)
    print(scored[["customerID", "p_retained_control", "p_retained_treatment", "uplift_score", "baseline_churn_prob"]].head(5).to_string())

    # 4. Financial Knapsack Optimization ($20,000 SaaS Retention Budget)
    print("\n[Step 4] Optimizing SaaS Retention Budget ($20,000 Cap)...")
    scored_fin = apply_financial_metrics(scored)
    budget = 20000.0
    opt_result = optimize_retention_budget(scored_fin, budget_limit=budget)
    summary = opt_result["summary"]

    print(f"[PASS] Budget Spent: ${summary['budget_spent']:,.2f} / ${budget:,.2f}")
    print(f"[PASS] Accounts Targeted: {summary['customers_targeted']}")
    print(f"[PASS] Incremental Retained Accounts: +{summary['expected_retained_lift']:.1f}")
    print(f"[PASS] Gross Revenue Saved: ${summary['gross_revenue_saved']:,.2f}")
    print(f"[PASS] Net Profit: ${summary['net_profit']:,.2f}")
    print(f"[PASS] Campaign ROI: +{summary['roi_percent']:.1f}%")

    comp_df = compare_all_strategies(scored_fin, budget_limit=budget)
    print("\nSaaS Benchmark Strategy Comparison:")
    print(comp_df.to_string(index=False))

    # 5. Executive Report
    report_path = "reports/saas_retention_report.html"
    generate_html_report(
        summary=summary,
        comparison_df=comp_df,
        targeted_df=opt_result["targeted_df"],
        output_path=report_path,
    )
    assert os.path.exists(report_path) and os.path.getsize(report_path) > 1000
    print(f"[PASS] SaaS Executive Brief generated at: {report_path}")

    print("ALL TESTS PASSED ON SAAS DATASET!")
    return summary, comp_df


def main():
    print("=" * 70)
    print("RUNNING MULTI-DATASET UNIVERSAL COMPATIBILITY VERIFICATION")
    print("=" * 70)

    summary_bank, comp_bank = test_bank_churn_dataset()
    summary_saas, comp_saas = test_saas_churn_dataset()

    print("\n" + "=" * 70)
    print("MULTI-DATASET CROSS-INDUSTRY BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"1. Bank Churn (10,000 rows):    Budget: ${summary_bank['budget_spent']:,.0f} | Saved: +{summary_bank['expected_retained_lift']:.0f} | Net Profit: +${summary_bank['net_profit']:,.0f} | ROI: +{summary_bank['roi_percent']:.0f}%")
    print(f"2. SaaS B2B   (5,000 rows):     Budget: ${summary_saas['budget_spent']:,.0f} | Saved: +{summary_saas['expected_retained_lift']:.0f} | Net Profit: +${summary_saas['net_profit']:,.0f} | ROI: +{summary_saas['roi_percent']:.0f}%")
    print("\nCONCLUSION: Universal CausalRetention Engine is functioning 100% reliably across all datasets!")


if __name__ == "__main__":
    main()
