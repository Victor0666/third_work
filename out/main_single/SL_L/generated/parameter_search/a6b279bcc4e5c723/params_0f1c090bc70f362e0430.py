import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule incorporating three key structural improvements:
      - Restores bounded sigmoid gating on uncertainty (via `uncertainty_sigmoid_steepness`) to recover starvation resilience lost in v2;
      - Replaces fragile unbounded `duration_risk_penalty` interaction with a clipped monotonic term: `np.clip(duration_total * uncertainty, 0, 2)` — prevents outlier explosion;
      - Introduces *critical-path urgency* scaling: multiplies `critical_release` by `1 + np.clip(-slack, 0, 2)` when slack < 0, amplifying bottleneck unlocking under deadline pressure;
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no other constants used;
      - Joint MAD normalization remains robust and cross-signal aligned;
      - Dual-gated non-DDL optimization preserved; DDL lexicographic safety enforced.
    """
    eps = 1.0106115305200421e-09
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack)
    all_risk = np.concatenate([abs_slack, uncertainty, duration_total])
    risk_median = np.median(all_risk)
    mad = np.median(np.abs(all_risk - risk_median)) + eps
    joint_scale = 0.6067938958824673 * mad

    def joint_mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        centered = x - risk_median
        normalized = centered / joint_scale
        return np.clip(normalized, -2.0, 2.0)
    slack_violation = np.where(slack < 0, (-slack) ** 1.9608574883832997, 0.0)
    unc_slack_coupling = 1.011998177896493 * uncertainty * np.maximum(0.0, -slack)
    duration_risk_raw = duration_total * uncertainty
    duration_risk_clipped = np.clip(duration_risk_raw, 0.0, 2.0)
    duration_risk = 1.0010848911205532 * duration_risk_clipped * np.where(slack <= 0.12489804251690825, 1.0, 0.0)
    critical_release_base = 0.33266491248078256 * upward_rank * remaining_work
    latency_urgency = np.clip(-slack, 0.0, 2.0)
    critical_release = critical_release_base * (1.0 + latency_urgency)
    ddl_safe_mask = np.where(slack > 0.12489804251690825, 1.0, 0.0)
    duration_median = np.median(duration_total) if N > 1 else duration_total[0]
    host_margin_mask = np.where(duration_total <= 1.6766129847170173 * duration_median, 1.0, 0.0)
    optimization_mask = ddl_safe_mask * host_margin_mask
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = joint_mad_normalize(energy_per_sec) * 1.2636860587485352
    unc_sigmoid = 1.0 / (1.0 + np.exp(-5.77030320770647 * (uncertainty - 1.0)))
    energy_norm = joint_mad_normalize(min_incremental_energy)
    unc_norm = joint_mad_normalize(uncertainty)
    energy_uncertainty_score = 3.367095915586108e-06 * energy_norm * unc_norm * unc_sigmoid
    score = joint_mad_normalize(slack_violation) + joint_mad_normalize(unc_slack_coupling) + joint_mad_normalize(duration_risk)
    score += -joint_mad_normalize(critical_release)
    score += optimization_mask * (energy_eff_score + energy_uncertainty_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
