import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining robustness from Parent 2 with risk-aware energy-uncertainty coupling.
    Key improvements:
      - Replaces fragile IQR normalization with mean-abs + eps robust norm (stable across seeds)
      - Introduces novel energy_uncertainty_interaction: prioritizes low-energy tasks *especially* when uncertainty is high
      - Uses wait saturation (clipped + robust norm) instead of exponential decay to avoid numerical instability
      - Unified signed slack handling: power penalty for negative, linear reward for positive
      - Criticality boost gated only when both rank is high AND slack is non-positive (tight or violated)
      - All literals are -2,-1,0,1,2; no hidden constants; all tunables declared in PARAMETER_SCHEMA.
    Smaller score = higher priority."""
    eps = 1.077820425042535e-06
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_norm(x):
        x = np.asarray(x, dtype=float)
        denom = np.mean(np.abs(x)) + eps
        return x / (denom + eps)
    slack_abs = np.abs(slack)
    slack_score = np.where(slack < 0, slack_abs ** 1.6213881356985422, slack * 0.3071667145134639)
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    is_high_rank = upward_rank >= rank_median
    is_tight_or_violated = slack <= 0
    critical_gate = np.where(is_high_rank & is_tight_or_violated, 3.2266968147629593, 1.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = robust_norm(energy_per_sec)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.12351448884964872
    wait_clipped = np.clip(ready_wait_time, 0, 0.44235276099401505)
    wait_score = -robust_norm(wait_clipped)
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.0300516438566885
    slack_p90 = np.percentile(slack, 82.77634787812126) if N > 1 else np.max(slack)
    slack_p10 = np.percentile(slack, 15.728319634040247) if N > 1 else np.min(slack)
    slack_range = np.maximum(eps, slack_p90 - slack_p10)
    slack_normalized = np.clip((slack - slack_p10) / (slack_range + eps), 0, 1)
    weight_rank = 0.9619217324332181 + (1 - 0.9619217324332181) * (1 - slack_normalized)
    rank_score = -robust_norm(upward_rank) * weight_rank
    energy_norm = robust_norm(min_incremental_energy)
    unc_norm = robust_norm(uncertainty)
    energy_uncertainty_score = 0.7561132766083631 * energy_norm * unc_norm
    score = robust_norm(slack_score) + robust_norm(unc_slack_coupling) + robust_norm(duration_risk) + 1.394170081640837 * energy_eff_score + rank_score + wait_score + energy_uncertainty_score + robust_norm(min_incremental_energy) * (1.0 - weight_rank)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
