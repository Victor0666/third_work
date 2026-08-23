import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with joint MAD normalization:
      - Joint MAD computed over |slack|, uncertainty, and duration_total for coherent risk alignment
      - All normalized features use same joint scale — eliminates mis-scaled interactions
      - Criticality signal uses jointly normalized rank×work product, gated only on violation + high-rank/work
      - Sigmoid gates use jointly normalized inputs (e.g., norm_slack) for stable gradients
      - Anti-starvation decay uses unitless norm_slack → no external time constant needed
      - No unused parameters; all PARAMS references matched and consumed
      - Numeric literals strictly limited to {-2,-1,0,1,2}
    """
    eps = 1.838560523362487e-06
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
    joint_features = np.stack([np.abs(slack), uncertainty, duration_total], axis=0).flatten()
    joint_med = np.median(joint_features)
    joint_mad = np.median(np.abs(joint_features - joint_med)) + eps
    joint_scale = 1.9093231827806443 * joint_mad

    def joint_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        centered = x - joint_med
        normalized = centered / joint_scale
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.332417802451157, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.0797071693505704
    duration_risk = duration_total * uncertainty * 1.3077735549000353
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= 0.4999833501757608 * rank_median
    is_high_work = remaining_work >= 0.4999833501757608 * work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 1.0, 0.0)
    rank_work_product = upward_rank * remaining_work
    critical_score = joint_normalize(rank_work_product) * critical_gate * 1.9267272407482636
    norm_slack = joint_normalize(slack)
    slack_gate = 1.0 / (1.0 + np.exp(-5.738666452398502 * norm_slack))
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = joint_normalize(energy_per_sec)
    rank_score = -joint_normalize(upward_rank) * (0.6774680624829659 * (1.0 - slack_gate) + (1.0 - 0.6774680624829659) * slack_gate)
    energy_norm = joint_normalize(min_incremental_energy)
    unc_norm = joint_normalize(uncertainty)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-5.738666452398502 * (unc_norm - 0.0)))
    energy_uncertainty_score = 0.6689181325101784 * energy_norm * unc_norm * unc_sigmoid
    wait_decay = np.exp(-np.abs(norm_slack))
    wait_score = joint_normalize(ready_wait_time) * wait_decay
    score = joint_normalize(slack_score) + joint_normalize(unc_slack_coupling) + joint_normalize(duration_risk)
    score += slack_gate * (1.1572602742884504 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score += critical_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
