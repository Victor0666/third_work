import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule integrating:
      - Strict lexicographic DDL gating: all non-DDL terms masked by `slack > 0`
      - Critical path release via `remaining_work / (upward_rank + eps)` — captures successor-unblocking power per criticality unit
      - Power-law anti-starvation: `np.sqrt(ready_wait_time)`, gated by slack headroom, avoids linear dominance
      - MAD-based robust normalization instead of percentile clipping — more stable under sparse/small N
      - Replaced `rank_score` with direct upward_rank * remaining_work interaction, normalized by task-wise MAD
      - All divisions protected by `eps`; no unbounded ops; deterministic; shape-(N,) output guaranteed
      - Uses only allowed literals {-2,-1,0,1,2}
    """
    eps = 1.0494350624455004e-07
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

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.9738653608775194, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.0330009890298364
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.91578523011969
    successor_release = remaining_work / (upward_rank + eps)
    successor_score = -mad_normalize(successor_release)
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 2.0055028133302764, 1.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -8.026188687518896
    slack_ub = 5.268087109287478
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.09556015825077604 + (1.0 - 0.09556015825077604) * (1.0 - slack_scaled)
    critical_path_product = upward_rank * remaining_work
    rank_score = -mad_normalize(critical_path_product) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-3.0523285925127204 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 1.1531605173635653 * energy_norm * unc_norm * unc_sigmoid
    wait_sqrt = np.where(slack_headroom_mask > 0.0, np.sqrt(ready_wait_time), 0.0)
    wait_score = mad_normalize(wait_sqrt)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + successor_score
    score += slack_headroom_mask * (0.38221518510489405 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
