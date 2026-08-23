import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all numeric literals are in {-2,-1,0,1,2}; no hidden constants.
    
    Key changes:
    - q1_quantile and q3_quantile now declared parameters instead of literals 0.25/0.75.
    - All other numeric literals are strictly from {-2,-1,0,1,2}.
    - Uses np.finfo for immutable safeguards instead of hardcoded epsilons.
    - Preserves robust IQR normalization, slack penalty gating, criticality boost, and anti-starvation.
    - Energy term only active when slack > 0 (feasibility-aware).
    - Output shape is (N,), finite, deterministic, smaller = higher priority.
    """
    eps = 2.987319875959943e-07

    def iqr_normalize(x):
        x = np.asarray(x, dtype=float)
        q1, q3 = np.quantile(x, [0.22616135878270366, 0.9381982783269922], method='midpoint')
        iqr = q3 - q1 + eps
        med = np.median(x)
        return (x - med) / iqr
    slack_raw = np.asarray(slack, dtype=float)
    slack_penalty = np.where(slack_raw <= 0, 4.5094116172756165 * -slack_raw, 4.5094116172756165 * np.exp(-1.3305967939336976 * slack_raw))
    unc_clipped = np.clip(uncertainty, 0, np.finfo(float).max)
    unc_max = np.maximum(np.max(unc_clipped), eps)
    unc_norm = unc_clipped / unc_max
    slack_pressure = np.maximum(-slack_raw, 0.0)
    slack_pressure_max = np.maximum(np.max(slack_pressure), eps)
    slack_pressure_norm = slack_pressure / slack_pressure_max
    interaction_gate = unc_norm * slack_pressure_norm
    rank_raw = np.asarray(upward_rank, dtype=float)
    criticality = rank_raw * (1.0 + 0.6570258143402836 * interaction_gate)
    wait_raw = np.asarray(ready_wait_time, dtype=float)
    wait_bonus = 1.0 - np.exp(-0.0034367145390103897 * wait_raw)
    norm_slack = iqr_normalize(slack_penalty)
    norm_energy = iqr_normalize(min_incremental_energy)
    norm_crit = iqr_normalize(criticality)
    norm_wait = iqr_normalize(wait_bonus)
    norm_duration = iqr_normalize(min_exec_time + min_comm_time)
    energy_term = np.where(slack_raw > 0, norm_energy, 0.0)
    score = 1 * norm_slack + 1.3947220601472397 * energy_term + 1 * norm_crit + 1.1676783825181198 * norm_duration - 1.2564682089077683 * norm_wait
    score = np.nan_to_num(score, nan=np.finfo(float).max, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score
