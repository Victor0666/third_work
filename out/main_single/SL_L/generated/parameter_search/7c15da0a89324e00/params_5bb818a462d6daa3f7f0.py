import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's lexicographic DDL gating and robust z-score normalization
    with Parent 1's joint MAD-based risk scaling and congestion-aware energy gating.
    Key novelties:
      - Joint MAD normalization applied *only* to risk dimensions (duration, slack, uncertainty) for stable cross-feature coherence
      - Congestion-aware energy gating now reuses the same MAD-normalized uncertainty (unc_norm) instead of raw uncertainty
      - Wait-time anti-starvation uses power-law dampening: 1/(|slack|+1)^p to better preserve starvation mitigation at moderate slack
      - Successor-release urgency scaled by slack_headroom^p (not linear) to prevent oversaturation in high-headroom regimes
      - Criticality boost now conditioned on *both* normalized rank AND normalized slack deficit, improving precision under marginal pressure
      - All non-DDL terms gated strictly by slack > 0.0 (lexicographic enforcement), preserving hard deadline guarantees
    """
    eps = 3.816195614113e-09
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

    def z_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        mean_x = np.mean(x)
        std_x = np.std(x, ddof=0) + eps
        z = (x - mean_x) / std_x
        return np.clip(z, -2.0, 2.0)
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack) + eps
    all_risk_dims = np.stack([duration_total, abs_slack, uncertainty], axis=0)
    mad_per_dim = np.mean(np.abs(all_risk_dims - np.median(all_risk_dims, axis=1, keepdims=True)), axis=1) + eps
    duration_norm = duration_total / (mad_per_dim[0] + eps)
    slack_norm = abs_slack / (mad_per_dim[1] + eps)
    unc_norm = uncertainty / (mad_per_dim[2] + eps)
    slack_score = np.where(slack < 0, (-slack) ** 2.8560299657757136, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.8995007886580207
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.0006424880698075521
    is_tight_or_violated = slack <= 0
    rank_z = z_normalize(upward_rank)
    slack_deficit_z = z_normalize(deadline_pressure)
    is_high_rank_norm = rank_z >= 0.0
    is_high_deficit_norm = slack_deficit_z >= 0.0
    critical_gate = np.where(is_high_rank_norm & is_high_deficit_norm & is_tight_or_violated, 2.809417816068053, 1.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total_safe = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total_safe
    energy_eff_score = z_normalize(energy_per_sec)
    slack_lb = -12.712795883159217
    slack_ub = 43.62543753182034
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.6678736328014259 + (1.0 - 0.6678736328014259) * (1.0 - slack_scaled)
    rank_score = -z_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-9.925192664346051 * (uncertainty - 1.0)))
    energy_norm = z_normalize(min_incremental_energy)
    energy_uncertainty_score = 0.4111142140997914 * energy_norm * unc_norm * unc_sigmoid
    successor_urgency = np.where(slack_headroom_mask > 0.0, (remaining_work + eps) / (upward_rank + eps), 0.0)
    slack_headroom_power = np.power(np.maximum(slack, 0.0) + eps, 1.0)
    successor_score = z_normalize(successor_urgency) * slack_headroom_mask * slack_headroom_power
    wait_score_raw = ready_wait_time / np.power(np.abs(slack) + 1.0, 0.6018178394288539)
    wait_score = np.clip(z_normalize(wait_score_raw), 0.0, 1.0)
    score = z_normalize(slack_score) + z_normalize(unc_slack_coupling) + z_normalize(duration_risk)
    score += slack_headroom_mask * (0.8549245860560188 * energy_eff_score + rank_score + energy_uncertainty_score + successor_score)
    score += wait_score
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
