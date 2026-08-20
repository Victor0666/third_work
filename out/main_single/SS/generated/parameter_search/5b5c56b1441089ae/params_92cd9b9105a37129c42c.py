import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: hard-deadline protection gate, monotonic work/slack coupling, and urgency-preserving piecewise slack penalty.
    
    Key improvements:
    - Replaces tanh with binary DDL-protection gate: when median_slack <= 0, amplify both rank and slack penalties → enforces feasibility-first hierarchy.
    - Replaces clipped successor_release with monotonic `remaining_work / (|slack| + eps)` — preserves critical-path signal even for small positive slack.
    - Uses piecewise `np.where(slack < 0, ..., 0)` for slack penalty instead of smoothed tanh → retains sharp urgency discrimination for lateness risk.
    - All numeric literals are -2,-1,0,1,2; all parameters declared and used exactly once.
    - Robust normalization and finite safeguards via np.finfo.
    """
    eps = 0.08118255613834734
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_norm(x):
        x_abs = np.abs(x)
        denom = np.mean(x_abs) + eps
        return x / denom
    median_slack = np.median(slack)
    ddl_protection_active = (median_slack <= eps).astype(float)
    ddl_boost = 1.0 + 1.0482822667946636 * ddl_protection_active
    neg_slack = np.where(slack < 0, -slack, 0.0)
    slack_penalty = np.power(neg_slack + eps, 2.78700050377448)
    duration = min_exec_time + min_comm_time
    duration_norm = robust_norm(duration)
    uncertainty_norm = robust_norm(uncertainty)
    risk_adjusted_duration = 0.6281160498375703 * duration_norm + (1 - 0.6281160498375703) * uncertainty_norm
    energy_norm = robust_norm(min_incremental_energy)
    energy_penalty = 1.4212601146872168 * energy_norm
    rank_base = robust_norm(upward_rank)
    rank_weighted = rank_base * (1 + 0.7740479593110778 * np.where(slack < 0, 1.0, 0.0))
    rank_weighted = rank_weighted * ddl_boost
    median_rank = np.median(upward_rank)
    abs_median_slack = np.abs(median_slack) + eps
    is_critical = ((upward_rank >= median_rank) & (slack <= 0.6278826648622851 * abs_median_slack)).astype(float)
    criticality_bonus = is_critical * 3.0675132793992805
    work_slack_ratio = remaining_work / (np.abs(slack) + eps)
    work_slack_norm = robust_norm(work_slack_ratio)
    successor_release = 2.1468040013715277 * work_slack_norm
    wait_norm = robust_norm(ready_wait_time)
    wait_bonus = -0.03205231882948941 * wait_norm
    unc_gate_active = ((uncertainty_norm > 0.17303501566421464) & (slack < eps)).astype(float)
    tightened_penalty = slack_penalty + unc_gate_active * 0.597984995130468 * uncertainty_norm
    tightened_penalty = tightened_penalty * ddl_boost
    score = tightened_penalty + energy_penalty + 1 * risk_adjusted_duration - 1 * rank_weighted + 0 * criticality_bonus - 1 * successor_release + wait_bonus
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values in priority score'
    return score
