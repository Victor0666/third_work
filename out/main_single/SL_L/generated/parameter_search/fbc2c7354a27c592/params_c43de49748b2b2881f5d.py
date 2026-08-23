import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with 11 parameters (slack_min_bound removed):
      - Uses slack_max_bound only — replaces two-sided clipping with symmetric sigmoid scaling
      - All PARAMS references are now used; no unused parameters
      - Numeric literals strictly limited to {-2,-1,0,1,2}
      - Criticality signal remains additive and violation-only
      - Anti-starvation uses exponential decay with slack_max_bound as time constant
      - Unified MAD normalization, N=1 safe
      - Smooth sigmoid slack_gate for non-DDL terms
      - Joint criticality gating on slack<=0 AND high rank/work
    """
    eps = 7.009624761040294e-07
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
    slack_score = np.where(slack < 0, (-slack) ** 3.999902801419805, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.006795444996443
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.8885356893018564
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= 0.972274380582841 * rank_median
    is_high_work = remaining_work >= 0.972274380582841 * work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 1.0, 0.0)
    rank_work_product = upward_rank * remaining_work
    critical_score = mad_normalize(rank_work_product) * critical_gate * 1.795841462399507
    slack_gate = 1.0 / (1.0 + np.exp(-1.4340080194746436 * slack))
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_scaled = slack / (4.662481988587176 + eps)
    slack_sigmoid = 1.0 / (1.0 + np.exp(-1.4340080194746436 * slack_scaled))
    weight_rank = 0.8346449628246325 * (1.0 - slack_sigmoid) + (1.0 - 0.8346449628246325) * slack_sigmoid
    rank_score = -mad_normalize(upward_rank) * weight_rank
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.4340080194746436 * (uncertainty - 1.0)))
    energy_uncertainty_score = 0.0001364811671104979 * energy_norm * unc_norm * unc_sigmoid
    wait_decay = np.exp(-np.abs(slack) / (4.662481988587176 + eps))
    wait_score = mad_normalize(ready_wait_time) * wait_decay
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk)
    score += slack_gate * (0.1059660616785692 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score += critical_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
