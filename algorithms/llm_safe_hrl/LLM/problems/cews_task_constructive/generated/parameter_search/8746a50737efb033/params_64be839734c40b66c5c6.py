import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining best structural elements from both parents:
      - Joint DDL-risk gate: activated when (slack <= median_slack * ratio) AND (normalized_uncertainty > threshold)
      - Uncertainty-scaled successor-release penalty: penalizes tasks whose successors have high remaining_work but low slack, amplified by uncertainty under joint risk.
      - Critical-path pressure: upward_rank × remaining_work, gated only under joint DDL+uncertainty risk.
      - Uncertainty-modulated energy penalty: marginal energy scaled by (1 + coupling * uncertainty), preserving direct physical interpretation.
      - Bounded power-law wait saturation (not decay): monotonic fairness boost with robust clipping and MAD-free normalization.
      - All normalizations use median/ptp with epsilon fallbacks; no std-based dispersion measures to avoid NaN on degenerate inputs.
      - Final score preserves hard-deadline dominance: neg_slack >> gated penalties >> energy >> wait_saturation.
      - Exactly one conditional branch (DDL gate); all operations vectorized and finite.
    """
    eps = 1.0029748806478046e-06
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
    high_risk_slack_condition = slack <= median_slack * 0.29045521042828326 + eps
    unc_min = np.min(uncertainty) if N > 0 else 0.0
    unc_max = np.max(uncertainty) if N > 0 else eps
    normalized_uncertainty = (uncertainty - unc_min) / (unc_max - unc_min + eps)
    high_uncertainty_condition = normalized_uncertainty > 0.19111445584299158
    ddl_risk_gate = np.where(high_risk_slack_condition & high_uncertainty_condition, 1.0, 0.0)
    critical_path_pressure = upward_rank * remaining_work
    gated_critical_pressure = ddl_risk_gate * 0.36309720747100027 * critical_path_pressure
    slack_feasibility = np.clip(slack / (np.abs(median_slack) + eps), 0.0, 1.0)
    slack_feasibility_sharpened = np.power(slack_feasibility + eps, 0.7587501505258318)
    successor_release_penalty = remaining_work * (1.0 - slack_feasibility_sharpened)
    gated_successor_penalty = ddl_risk_gate * successor_release_penalty * 1.8468768640690925 * (1.0 + 1.6307043878160756 * normalized_uncertainty)
    energy_penalty = min_incremental_energy * (1.0 + 0.6440504008613187 * uncertainty)
    wait_base = np.maximum(np.mean(ready_wait_time), eps) if N > 0 else eps
    wait_scaled = np.clip(ready_wait_time / (wait_base + eps), 0.0, 2.0)
    wait_saturation = np.power(wait_scaled + eps, 1.1465124567651606)
    score = neg_slack + gated_successor_penalty + gated_critical_pressure + energy_penalty - wait_saturation
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
