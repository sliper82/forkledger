from __future__ import annotations

from .models import ForkRecord


def compute_regret(record: ForkRecord) -> dict[str, float]:
    """
    Regret is measured as:
        best_estimated_value - branch_value

    For the chosen branch, branch_value is the realized value.
    For alternative branches, branch_value is the estimated counterfactual value.
    Lower regret is better.
    """
    estimates = {item.branch_name: item.estimated_value for item in record.estimated_outcomes}
    branch_values = dict(estimates)
    branch_values[record.chosen_branch] = record.realized_value

    if not branch_values:
        return {}

    best_value = max(branch_values.values())
    return {branch: round(best_value - value, 6) for branch, value in branch_values.items()}


def fill_regret(record: ForkRecord) -> ForkRecord:
    record.regret_vector = compute_regret(record)
    return record
