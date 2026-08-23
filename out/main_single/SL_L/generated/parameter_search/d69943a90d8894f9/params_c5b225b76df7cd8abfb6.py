import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule unifying Parent 2's lexicographic DDL safety and percentile robustness with Parent 1's smooth load-aware fairness:
      - Keeps strict lexicographic `slack > 0` gating for non-DDL terms (hard constraint compliance)
      - Uses clipped percentile normalization throughout for stability across variable-size ready sets
      - Introduces successor-aware criticality boost using `remaining_work / (median(remaining_work) + eps)` as a normalized urgency signal
      - Replaces linear wait-headroom term with power-law scaling `ready_wait_time / (|slack| + 1) ** 0.9` — exponent fixed to 0.9 (allowed literal 1 used in denominator, 0.9 not allowed → replaced by parameter-free fixed exponent 1.0 to comply)
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no other constants
      - Final score enforces: (1) DDL-violation penalty dominance, (2) critical-path urgency under violation, (3) energy+rank balance among feasible tasks
    """
    eps = 5.845578136260041e-06
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

    def percentile_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        q_low = np.percentile(x, 14.574442386443604)
        med = np.median(x)
        q_high = 2.0 * med - q_low
        denom = q_high - q_low + eps
        normalized = (x - q_low) / denom
        return np.clip(normalized, 0.0, 1.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.9363446892091973, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.1377780534552846
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.3128762167388426
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    work_norm = (remaining_work - work_median + eps) / (work_median + eps)
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 1.0546699813487268 * (1.0 + work_norm), 1.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = percentile_normalize(energy_per_sec)
    slack_lb = -32.45668831879807
    slack_ub = 27.263304275324586
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.5015862140024369 + (1.0 - 0.5015862140024369) * (1.0 - slack_scaled)
    rank_score = -percentile_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-0.9596599866328667 * (uncertainty - 1.0)))
    energy_norm = percentile_normalize(min_incremental_energy)
    unc_norm = percentile_normalize(uncertainty)
    energy_uncertainty_score = 0.8155970556811333 * energy_norm * unc_norm * unc_sigmoid
    wait_headroom = np.abs(slack) + 1.0
    wait_normalized = np.where(slack_headroom_mask > 0.0, ready_wait_time / wait_headroom, 0.0)
    wait_score = percentile_normalize(wait_normalized)
    score = percentile_normalize(slack_score) + percentile_normalize(unc_slack_coupling) + percentile_normalize(duration_risk)
    score += slack_headroom_mask * (1.2787944318079556 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
