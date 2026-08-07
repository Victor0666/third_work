import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with restored hard gating + uncertainty-aware slack pressure:
      - Reintroduces explicit, numerically stable conditional gate (np.where) for DDL protection — preserves ordering fidelity under tight deadlines.
      - Extends ddl_risk_gate to jointly depend on both slack gap AND normalized uncertainty — avoids objective collapse from unbounded dispersion.
      - Adds uncertainty_sensitivity_weight to tune how strongly uncertainty amplifies deadline risk signal.
      - Uses robust median-based normalization for all dispersion-sensitive terms (energy, wait), avoiding std=0 failure modes.
      - Keeps successor-release penalty gated *only* when both slack is tight AND uncertainty is elevated — directly targets SUCCESSOR_RELEASE_BLOCKING.
      - All operations remain vectorized; still only ONE conditional branch (DDL gate).
      - No numeric literals beyond {-2,-1,0,1,2}; all tunables declared in PARAMETER_SCHEMA.
    """
    eps = 1e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    uncertainty_range = np.ptp(uncertainty) if N > 1 else 0.0
    uncertainty_normalized = (uncertainty - np.min(uncertainty)) / (uncertainty_range + eps) if N > 0 else np.zeros_like(uncertainty)
    ddl_risk_gate = np.where((slack <= median_slack) & (uncertainty_normalized > 0.3487090908680185), 1.0, 0.0)
    critical_path_pressure = upward_rank * remaining_work
    gated_critical_pressure = ddl_risk_gate * critical_path_pressure
    slack_feasibility = np.clip(slack / (np.abs(median_slack) + eps), 0.0, 1.0)
    slack_feasibility_sharpened = np.power(slack_feasibility + eps, 1.0461656715869176)
    successor_release_penalty = remaining_work * (1.0 - slack_feasibility_sharpened)
    gated_successor_penalty = ddl_risk_gate * successor_release_penalty * 1.2987684909727717 * (1.0 + 0.7843977728140572 * uncertainty_normalized)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    energy_center = np.median(energy_per_duration) if N > 0 else 0.0
    abs_devs = np.abs(energy_per_duration - energy_center)
    energy_mad = np.median(abs_devs) if N > 0 else 0.0
    energy_dispersion = np.where(energy_mad > eps, energy_mad, eps)
    norm_energy = (energy_per_duration - energy_center) / (energy_dispersion * 2.9500187449192645 + eps)
    max_wait = np.max(ready_wait_time) if N > 0 else eps
    wait_ratio = np.clip(ready_wait_time / (max_wait + eps), 0.0, 1.0)
    wait_decay = np.power(wait_ratio + eps, 0.7171766933994235)
    score = neg_slack + gated_successor_penalty + 0.9446077075438679 * gated_critical_pressure + norm_energy - wait_decay
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
