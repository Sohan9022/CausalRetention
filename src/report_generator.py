import datetime
import os
import pandas as pd


def generate_html_report(
    summary: dict,
    comparison_df: pd.DataFrame,
    targeted_df: pd.DataFrame,
    output_path: str = "reports/executive_retention_report.html",
) -> str:
    """
    Generates a professional, board-ready HTML Executive Brief.
    Can be opened directly in any browser or printed/saved as PDF (Ctrl + P).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%B %d, %Y - %H:%M:%S")

    # Format sample targeted table (top 10) dynamically based on available columns
    core_cols = ["customerID", "uplift_score", "campaign_offer_cost", "expected_net_profit"]
    all_cols = list(targeted_df.columns)
    
    # Identify up to 3 domain features present in the dataset
    excluded = set(core_cols + [
        "quadrant", "retained", "churn", "treatment", "annual_clv", "offer_cost",
        "campaign_clv", "expected_gross_revenue_saved", "profit_per_dollar_spent",
        "p_retained_treatment", "p_retained_control", "baseline_churn_prob"
    ])
    domain_features = [c for c in all_cols if c not in excluded][:3]

    selected_cols = ["customerID"] + domain_features + ["uplift_score", "campaign_offer_cost", "expected_net_profit"]
    top_accounts = targeted_df.head(10)[selected_cols].copy()

    # Create readable column names
    col_rename = {
        "customerID": "Customer ID",
        "uplift_score": "Uplift Score (Tau)",
        "campaign_offer_cost": "Offer Cost ($)",
        "expected_net_profit": "Expected Net Profit ($)",
    }
    for c in domain_features:
        col_rename[c] = c.replace("_", " ").title()

    top_accounts = top_accounts.rename(columns=col_rename)

    top_accounts_html = top_accounts.to_html(
        classes="table table-hover table-striped",
        index=False,
        border=0,
        float_format=lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x),
    )

    comp_table_html = comparison_df.to_html(
        classes="table table-bordered", index=False, border=0
    )

    # Breakdown badges
    quadrant_counts = summary.get("quadrant_breakdown", {})
    quadrant_items = "".join(
        [
            f"<li class='list-group-item d-flex justify-content-between align-items-center'>"
            f"<span>{k}</span><span class='badge bg-primary rounded-pill'>{v}</span></li>"
            for k, v in quadrant_counts.items()
        ]
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CausalRetention - Executive Retention Brief</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            background-color: #f8f9fa;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #212529;
            padding: 30px;
        }}
        .report-card {{
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
            padding: 30px;
            margin-bottom: 25px;
        }}
        .kpi-box {{
            background: #f1f5f9;
            border-left: 5px solid #0d6efd;
            border-radius: 8px;
            padding: 16px;
        }}
        .kpi-val {{
            font-size: 26px;
            font-weight: 700;
            color: #0d6efd;
        }}
        .kpi-lbl {{
            font-size: 13px;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 600;
        }}
        .badge-positive {{
            background-color: #d1e7dd;
            color: #0f5132;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: 600;
        }}
        .print-btn {{
            position: fixed;
            bottom: 25px;
            right: 25px;
            z-index: 1000;
        }}
        @media print {{
            .print-btn {{ display: none; }}
            body {{ padding: 0; background-color: #ffffff; }}
            .report-card {{ box-shadow: none; border: 1px solid #dee2e6; }}
        }}
    </style>
</head>
<body>
    <button class="btn btn-primary print-btn shadow" onclick="window.print()">🖨️ Print / Save as PDF</button>

    <div class="container">
        <!-- Header -->
        <div class="report-card">
            <div class="d-flex justify-content-between align-items-center border-bottom pb-3 mb-4">
                <div>
                    <h2 class="fw-bold mb-1" style="color: #0f172a;">Executive Retention & Uplift Decision Brief</h2>
                    <p class="text-muted mb-0">CausalRetention Decision Intelligence Engine</p>
                </div>
                <div class="text-end">
                    <span class="badge bg-success mb-1">Status: Approved for Dispatch</span><br>
                    <small class="text-muted">Generated: {timestamp}</small>
                </div>
            </div>

            <!-- KPI Row -->
            <div class="row g-3">
                <div class="col-md-3">
                    <div class="kpi-box">
                        <div class="kpi-lbl">Budget Utilized</div>
                        <div class="kpi-val">${summary['budget_spent']:,.2f}</div>
                        <small class="text-muted">Cap: ${summary['budget_limit']:,.2f}</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="kpi-box" style="border-left-color: #198754;">
                        <div class="kpi-lbl">Customers Saved</div>
                        <div class="kpi-val" style="color: #198754;">+{summary['expected_retained_lift']:.0f}</div>
                        <small class="text-muted">From {summary['customers_targeted']} targeted</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="kpi-box" style="border-left-color: #0dcaf0;">
                        <div class="kpi-lbl">Net Revenue Preserved</div>
                        <div class="kpi-val" style="color: #0dcaf0;">${summary['net_profit']:,.2f}</div>
                        <small class="text-muted">Gross: ${summary['gross_revenue_saved']:,.2f}</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="kpi-box" style="border-left-color: #ffc107;">
                        <div class="kpi-lbl">Campaign ROI</div>
                        <div class="kpi-val" style="color: #d39e00;">+{summary['roi_percent']:.1f}%</div>
                        <small class="text-muted">Net Return / Spend</small>
                    </div>
                </div>
            </div>
        </div>

        <!-- Strategy Benchmark Comparison -->
        <div class="report-card">
            <h4 class="fw-bold mb-3" style="color: #1e293b;">1. Strategic Benchmark Evaluation</h4>
            <p class="text-secondary">
                Comparative analysis demonstrating financial returns achieved under identical budget constraints ($<strong>{summary['budget_limit']:,.2f}</strong>).
                Notice how traditional churn modeling generates a net loss by subsidizing Lost Causes, while Causal Uplift isolates incremental persuadable accounts.
            </p>
            <div class="table-responsive">
                {comp_table_html}
            </div>
        </div>

        <!-- Behavioral Breakdown -->
        <div class="row">
            <div class="col-md-6">
                <div class="report-card h-100">
                    <h5 class="fw-bold mb-3" style="color: #1e293b;">2. Targeted Cohort Composition</h5>
                    <p class="text-muted small">
                        Breakdown of the targeted accounts selected by the optimization engine under the current budget threshold.
                    </p>
                    <ul class="list-group">
                        {quadrant_items}
                    </ul>
                </div>
            </div>
            <div class="col-md-6">
                <div class="report-card h-100">
                    <h5 class="fw-bold mb-3" style="color: #1e293b;">3. Operational Interventions</h5>
                    <div class="alert alert-success py-2">
                        <strong>Persuadables:</strong> Dispatch retention offer (20% discount or 3-month speed booster).
                    </div>
                    <div class="alert alert-secondary py-2">
                        <strong>Sure Things:</strong> Suppress from all promotional discounts. Organic retention probability exceeds 90%.
                    </div>
                    <div class="alert alert-warning py-2">
                        <strong>Sleeping Dogs:</strong> Blacklist from marketing notifications to avoid triggering dormant churn.
                    </div>
                </div>
            </div>
        </div>

        <!-- Priority Accounts Preview -->
        <div class="report-card">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h5 class="fw-bold mb-0" style="color: #1e293b;">4. Priority Dispatch List (Top 10 Accounts Sample)</h5>
                <span class="badge bg-secondary">Total Targeted: {summary['customers_targeted']}</span>
            </div>
            <div class="table-responsive">
                {top_accounts_html}
            </div>
            <p class="text-muted small mt-2 mb-0">
                * Complete target roster of all {summary['customers_targeted']} customers is available for direct export to CRM/marketing platforms via CSV.
            </p>
        </div>

        <!-- Footer -->
        <div class="text-center text-muted small py-3">
            Generated autonomously by <strong>CausalRetention v1.0</strong> • Powered by T-Learner XGBoost & Financial Knapsack Optimization.
        </div>
    </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Executive Report saved to {output_path}")
    return output_path


if __name__ == "__main__":
    from data_loader import load_or_create_data
    from optimizer import apply_financial_metrics, compare_all_strategies, optimize_retention_budget
    from uplift_model import TLearnerUpliftEngine

    engine = TLearnerUpliftEngine.load()
    data = load_or_create_data()
    scored = engine.predict_uplift(data)
    scored_fin = apply_financial_metrics(scored)

    opt = optimize_retention_budget(scored_fin, budget_limit=10000.0)
    comp = compare_all_strategies(scored_fin, budget_limit=10000.0)

    path = generate_html_report(opt["summary"], comp, opt["targeted_df"])
    print(f"Report test generated: {path}")
