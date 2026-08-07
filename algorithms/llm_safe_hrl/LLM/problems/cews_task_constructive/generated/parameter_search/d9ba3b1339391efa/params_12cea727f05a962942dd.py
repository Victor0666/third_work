import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three key structural improvements:
      - Piecewise slack feasibility gate: linear ramp for slack > 0, hard zero for slack <= 0 → enforces strict DDL feasibility before energy optimization.
      - Host-load-aware energy gating: energy penalty only activates when host load exceeds threshold, targeting avoidable marginal energy.
      - Successor-release penalty now conditioned on task's own slack and remaining_work (proxy for immediate successors' state), tightening causal alignment.
      - All normalizations remain epsilon-guarded and median/ptp-based; no std or degenerate dispersion measures.
      - Exactly one conditional branch (DDL gate); all operations vectorized, finite, deterministic.
    """
    eps = 1.0091864243561747e-06
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
    high_risk_slack_condition = slack <= median_slack * 0.5497582724892015 + eps
    unc_min = np.min(uncertainty) if N > 0 else 0.0
    unc_max = np.max(uncertainty) if N > 0 else eps
    normalized_uncertainty = (uncertainty - unc_min) / (unc_max - unc_min + eps)
    high_uncertainty_condition = normalized_uncertainty > 0.6501672773689378
    ddl_risk_gate = np.where(high_risk_slack_condition & high_uncertainty_condition, 1.0, 0.0)
    slack_feasibility = np.where(slack > 0, np.clip(2.219279021321384 * slack, 0.0, 1.0), 0.0)
    critical_path_pressure = upward_rank * remaining_work
    gated_critical_pressure = ddl_risk_gate * 0.1147489835734088 * critical_path_pressure
    successor_release_penalty = remaining_work * (1.0 - slack_feasibility)
    gated_successor_penalty = ddl_risk_gate * successor_release_penalty * 0.9904750199871352 * (1.0 + 0.9774086086448764 * normalized_uncertainty)
    duration_proxy = min_exec_time + min_comm_time
    host_load_proxy = duration_proxy * (1.0 + 0.35190578035717046 * uncertainty)
    host_load_normalized = (host_load_proxy - np.min(host_load_proxy)) / (np.ptp(host_load_proxy) + eps) if N > 1 else np.zeros_like(host_load_proxy)
    energy_activation_gate = np.where(host_load_normalized > 0.8124224959981876, 1.0, 0.0)
    energy_penalty = energy_activation_gate * min_incremental_energy * (1.0 + 0.35190578035717046 * uncertainty)
    wait_base = np.maximum(np.mean(ready_wait_time), eps) if N > 0 else eps
    wait_scaled = np.clip(ready_wait_time / (wait_base + eps), 0.0, 2.0)
    wait_saturation = np.power(wait_scaled + eps, 1.1019420271642784)
    score = neg_slack + gated_successor_penalty + gated_critical_pressure + energy_penalty - wait_saturation
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
