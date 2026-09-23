"""
Generate realistic synthetic PAIMANA-aligned project monitoring data
for the INFRAguard AI prototype.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

MINISTRIES = [
    "Ministry of Road Transport & Highways",
    "Ministry of Railways",
    "Ministry of Power",
    "Ministry of Jal Shakti",
    "Ministry of Housing & Urban Affairs",
    "Ministry of Civil Aviation",
    "Ministry of Ports, Shipping & Waterways",
    "Ministry of Petroleum & Natural Gas",
    "Ministry of Coal",
    "Ministry of Steel",
    "Ministry of Chemicals & Fertilizers",
    "Ministry of Heavy Industries",
    "Department of Telecommunications",
    "Ministry of Agriculture",
    "Ministry of Education",
    "Ministry of Health & Family Welfare",
    "Ministry of Rural Development",
]

SECTORS = [
    "Roads & Highways", "Railways", "Power Generation", "Power Transmission",
    "Irrigation", "Urban Development", "Airports", "Ports", "Oil & Gas",
    "Coal Mining", "Steel Plants", "Telecom", "Agriculture Infra",
    "Education Infra", "Health Infra", "Rural Roads", "Water Supply"
]

STATUSES = ["Ongoing", "Delayed", "Completed", "Stalled", "Under Review"]


def generate_projects(n: int = 200) -> pd.DataFrame:
    """Generate synthetic project portfolio similar to PAIMANA structure."""
    rows = []
    base_date = datetime(2024, 1, 1)

    for i in range(1, n + 1):
        ministry = np.random.choice(MINISTRIES)
        sector = np.random.choice(SECTORS)
        status = np.random.choice(STATUSES, p=[0.45, 0.25, 0.15, 0.08, 0.07])

        original_cost = np.round(np.random.lognormal(mean=5.5, sigma=1.2), 2)  # in Cr
        revised_cost = original_cost * np.random.uniform(1.0, 1.8)
        expenditure = revised_cost * np.random.uniform(0.15, 0.95)

        planned_duration = np.random.randint(12, 72)  # months
        start_date = base_date + timedelta(days=np.random.randint(0, 700))
        planned_end = start_date + timedelta(days=planned_duration * 30)

        physical_progress = np.clip(np.random.beta(2, 2) * 100, 5, 98)
        if status == "Completed":
            physical_progress = np.random.uniform(95, 100)
            expenditure = revised_cost * np.random.uniform(0.92, 1.05)

        # Milestone slippage (months)
        milestone_slip = max(0, np.random.normal(2.5, 4))
        if status == "Delayed" or status == "Stalled":
            milestone_slip = abs(np.random.normal(8, 5))

        # Spend deviation (%)
        expected_spend_ratio = physical_progress / 100
        actual_spend_ratio = expenditure / revised_cost
        spend_deviation = (actual_spend_ratio - expected_spend_ratio) * 100

        # Cost overrun %
        cost_overrun_pct = ((revised_cost - original_cost) / original_cost) * 100

        # Schedule delay probability drivers
        progress_variance = abs(physical_progress - (expenditure / revised_cost * 100))

        rows.append({
            "project_id": f"PRJ-{i:04d}",
            "project_name": f"{sector} Project {i}",
            "ministry": ministry,
            "sector": sector,
            "status": status,
            "original_cost_cr": round(original_cost, 2),
            "revised_cost_cr": round(revised_cost, 2),
            "expenditure_cr": round(expenditure, 2),
            "physical_progress_pct": round(physical_progress, 1),
            "planned_duration_months": planned_duration,
            "milestone_slip_months": round(milestone_slip, 1),
            "spend_deviation_pct": round(spend_deviation, 1),
            "cost_overrun_pct": round(cost_overrun_pct, 1),
            "progress_variance": round(progress_variance, 1),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "planned_end_date": planned_end.strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(rows)

    # Create target labels for training (binary risk flags)
    # High cost risk if overrun > 25% or spend deviation extreme
    df["cost_risk"] = ((df["cost_overrun_pct"] > 25) | (df["spend_deviation_pct"].abs() > 20)).astype(int)

    # High delay risk if milestone slip > 6 months or progress lagging
    df["delay_risk"] = ((df["milestone_slip_months"] > 6) | (df["progress_variance"] > 25)).astype(int)

    # Overall high risk
    df["overall_risk"] = ((df["cost_risk"] == 1) | (df["delay_risk"] == 1)).astype(int)

    return df


if __name__ == "__main__":
    df = generate_projects(250)
    df.to_csv("data/sample_projects.csv", index=False)
    print(f"Generated {len(df)} projects → data/sample_projects.csv")
    print(df[["cost_risk", "delay_risk", "overall_risk"]].mean())
