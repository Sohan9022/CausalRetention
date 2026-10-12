import numpy as np
import pandas as pd


def classify_behavioral_quadrant(row: pd.Series, uplift_threshold: float = 0.10) -> str:
    """
    Categorizes customers into the 4 behavioral uplift quadrants:
      1. Persuadables: High positive uplift (Worth intervening)
      2. Sure Things: High natural retention, low uplift (Don't waste discount)
      3. Lost Causes: Low natural retention, low uplift (Will leave anyway)
      4. Sleeping Dogs: Negative uplift (Intervention triggers churn)
    """
    tau = row["uplift_score"]
    p_ctrl = row["p_retained_control"]

    if tau < -0.05:
        return "Sleeping Dog (Do Not Contact)"
    elif tau >= uplift_threshold:
        return "Persuadable (High Priority)"
    elif p_ctrl >= 0.60:
        return "Sure Thing (Organic Retain)"
    else:
        return "Lost Cause (Unresponsive)"


def apply_financial_metrics(
    df: pd.DataFrame,
    default_clv: float = None,
    default_offer_cost: float = None,
    uplift_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Computes expected financial return metrics for each customer based on their uplift score.
    """
    df = df.copy()

    # Assign quadrant
    df["quadrant"] = df.apply(lambda r: classify_behavioral_quadrant(r, uplift_threshold), axis=1)

    # Use existing or override CLV & offer cost
    if default_clv is not None:
        df["campaign_clv"] = default_clv
    else:
        if "annual_clv" in df.columns:
            df["campaign_clv"] = df["annual_clv"]
        elif "MonthlyCharges" in df.columns:
            df["campaign_clv"] = df["MonthlyCharges"] * 12
        else:
            df["campaign_clv"] = 600.0

    if default_offer_cost is not None:
        df["campaign_offer_cost"] = default_offer_cost
    else:
        df["campaign_offer_cost"] = df["offer_cost"] if "offer_cost" in df.columns else 40.0

    # Expected Incremental Revenue = Uplift * CLV
    df["expected_gross_revenue_saved"] = np.round(df["uplift_score"] * df["campaign_clv"], 2)

    # Expected Net Profit = (Uplift * CLV) - Offer_Cost
    df["expected_net_profit"] = np.round(
        df["expected_gross_revenue_saved"] - df["campaign_offer_cost"], 2
    )

    # Efficiency ratio: Profit per dollar of discount spend
    df["profit_per_dollar_spent"] = np.where(
        df["campaign_offer_cost"] > 0,
        df["expected_net_profit"] / df["campaign_offer_cost"],
        0.0,
    )

    return df


def optimize_retention_budget(
    df_scored: pd.DataFrame,
    budget_limit: float = 10000.0,
    strategy: str = "causal_uplift",
) -> dict:
    """
    Allocates a finite retention budget using one of three strategies:
      - 'causal_uplift': Targets highest expected net profit first (Causal retention).
      - 'traditional_churn': Targets highest baseline churn probability first (Standard ML).
      - 'random': Selects random customers (Random baseline).

    Returns detailed allocation metrics and list of targeted accounts.
    """
    df = df_scored.copy()

    # Sort based on chosen strategy
    if strategy == "causal_uplift":
        # Only consider candidates with positive expected net profit
        sorted_df = df.sort_values(by="expected_net_profit", ascending=False).reset_index(drop=True)
    elif strategy == "traditional_churn":
        sorted_df = df.sort_values(by="baseline_churn_prob", ascending=False).reset_index(drop=True)
    elif strategy == "random":
        sorted_df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Knapsack greedy budget allocation
    cum_cost = sorted_df["campaign_offer_cost"].cumsum()
    targeted_mask = cum_cost <= budget_limit

    targeted_df = sorted_df[targeted_mask].copy()
    non_targeted_df = sorted_df[~targeted_mask].copy()

    # Calculate aggregate financial outcomes
    budget_spent = targeted_df["campaign_offer_cost"].sum()
    customers_targeted = len(targeted_df)
    expected_retained_lift = targeted_df["uplift_score"].sum()
    gross_revenue_saved = targeted_df["expected_gross_revenue_saved"].sum()
    net_profit = targeted_df["expected_net_profit"].sum()
    roi_percent = (net_profit / budget_spent * 100) if budget_spent > 0 else 0.0

    summary = {
        "strategy": strategy,
        "budget_limit": budget_limit,
        "budget_spent": round(budget_spent, 2),
        "budget_remaining": round(budget_limit - budget_spent, 2),
        "customers_targeted": customers_targeted,
        "expected_retained_lift": round(expected_retained_lift, 1),
        "gross_revenue_saved": round(gross_revenue_saved, 2),
        "net_profit": round(net_profit, 2),
        "roi_percent": round(roi_percent, 1),
        "quadrant_breakdown": targeted_df["quadrant"].value_counts().to_dict(),
    }

    return {
        "summary": summary,
        "targeted_df": targeted_df,
        "non_targeted_df": non_targeted_df,
    }


def compare_all_strategies(df_scored: pd.DataFrame, budget_limit: float = 10000.0) -> pd.DataFrame:
    """Generates an executive comparison table across Causal Uplift vs Standard Churn vs Random."""
    res_causal = optimize_retention_budget(df_scored, budget_limit, strategy="causal_uplift")["summary"]
    res_churn = optimize_retention_budget(df_scored, budget_limit, strategy="traditional_churn")["summary"]
    res_random = optimize_retention_budget(df_scored, budget_limit, strategy="random")["summary"]

    comp = pd.DataFrame([
        {
            "Strategy": "Causal Uplift (Our Approach)",
            "Budget Spent": f"${res_causal['budget_spent']:,.2f}",
            "Customers Targeted": res_causal["customers_targeted"],
            "Customers Saved": f"{res_causal['expected_retained_lift']:.0f}",
            "Gross Revenue Saved": f"${res_causal['gross_revenue_saved']:,.2f}",
            "Net Profit Gained": f"${res_causal['net_profit']:,.2f}",
            "Campaign ROI": f"{res_causal['roi_percent']:.1f}%",
        },
        {
            "Strategy": "Standard Churn Prediction (Traditional)",
            "Budget Spent": f"${res_churn['budget_spent']:,.2f}",
            "Customers Targeted": res_churn["customers_targeted"],
            "Customers Saved": f"{res_churn['expected_retained_lift']:.0f}",
            "Gross Revenue Saved": f"${res_churn['gross_revenue_saved']:,.2f}",
            "Net Profit Gained": f"${res_churn['net_profit']:,.2f}",
            "Campaign ROI": f"{res_churn['roi_percent']:.1f}%",
        },
        {
            "Strategy": "Random Promotion (Untargeted)",
            "Budget Spent": f"${res_random['budget_spent']:,.2f}",
            "Customers Targeted": res_random["customers_targeted"],
            "Customers Saved": f"{res_random['expected_retained_lift']:.0f}",
            "Gross Revenue Saved": f"${res_random['gross_revenue_saved']:,.2f}",
            "Net Profit Gained": f"${res_random['net_profit']:,.2f}",
            "Campaign ROI": f"{res_random['roi_percent']:.1f}%",
        },
    ])
    return comp


if __name__ == "__main__":
    from uplift_model import TLearnerUpliftEngine
    from data_loader import load_or_create_data

    engine = TLearnerUpliftEngine.load()
    data = load_or_create_data()
    scored = engine.predict_uplift(data)
    scored_fin = apply_financial_metrics(scored)

    print("\nQuadrant Distribution across All Customers:")
    print(scored_fin["quadrant"].value_counts(normalize=True))

    comp_table = compare_all_strategies(scored_fin, budget_limit=10000.0)
    print("\nExecutive Strategy Comparison ($10,000 Budget):")
    print(comp_table.to_string(index=False))
