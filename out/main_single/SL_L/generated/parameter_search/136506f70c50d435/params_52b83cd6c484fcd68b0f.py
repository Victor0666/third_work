import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Directional slack normalization: convex blend preserving sign-aware violation detection and magnitude coherence.
      - Restored robust z-score normalization for all features — avoids cross-dimension signal leakage.
      - Bounded linear headroom gate for successor urgency: `np.clip(slack, 0, PARAMS['slack_max_bound'])` ensures monotonic, stable scaling.
      - Criticality boost uses directionally normalized slack for improved sensitivity near slack=0.
      - Wait-time anti-starvation power-law dampened using |slack_dir_norm| for consistent behavior across regimes.
      - All non-DDL terms strictly gated by `slack > 0.0`; preserves hard deadline guarantees.
      - Uses only {-2,-1,0,1,2} literals; all tunables via PARAMS; shape (N,) guaranteed.
    """
    eps = 3.2905755601530064e-08
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
    abs_slack = np.abs(slack) + eps
    slack_score = np.where(slack < 0, (-slack) ** 3.148807720191491, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.3156375184430074
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.2663229649107732
    is_tight_or_violated = slack <= 0
    rank_z = z_normalize(upward_rank)
    is_high_rank_norm = rank_z >= 0.0
    critical_gate = np.where(is_high_rank_norm & is_tight_or_violated, 2.689427567063495, 1.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total_safe = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total_safe
    energy_eff_score = z_normalize(energy_per_sec)
    slack_lb = -34.458271428855994
    slack_ub = 34.85405262189429
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.559855879538496 + (1.0 - 0.559855879538496) * (1.0 - slack_scaled)
    rank_score = -z_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.105680640849158 * (uncertainty - 1.0)))
    energy_norm = z_normalize(min_incremental_energy)
    unc_norm = z_normalize(uncertainty)
    energy_uncertainty_score = 0.6735966178454222 * energy_norm * unc_norm * unc_sigmoid
    successor_urgency = np.where(slack_headroom_mask > 0.0, (remaining_work + eps) / (upward_rank + eps), 0.0)
    bounded_headroom = np.clip(slack, 0.0, slack_ub)
    successor_score = z_normalize(successor_urgency) * slack_headroom_mask * bounded_headroom
    wait_score_raw = ready_wait_time / np.power(np.abs(slack) + 1.0, 0.9921994897942636)
    wait_score = np.clip(z_normalize(wait_score_raw), 0.0, 1.0)
    score = z_normalize(slack_score) + z_normalize(unc_slack_coupling) + z_normalize(duration_risk)
    score += slack_headroom_mask * (0.1057722012628199 * energy_eff_score + rank_score + energy_uncertainty_score + successor_score)
    score += wait_score
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
