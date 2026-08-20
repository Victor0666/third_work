import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's hard DDL gate and successor-release logic with Parent 1's negative-slack-only activation and robust median-MAD normalization.
       Introduces novel *dual-gated criticality*: upward_rank boosted only under slack<=0 (hard) AND via smooth sigmoid (soft), weighted by uncertainty.
       Uses unified robust normalization; removes unstable power penalties; enforces deterministic shape and finite output."""
    eps = 0.011197149617130456
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 0:
            return np.zeros(0, dtype=float)
        center = np.median(x) if N > 1 else x[0]
        dev = x - center
        spread = np.median(np.abs(dev)) if N > 1 else np.abs(dev[0]) + eps
        return dev / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    ddl_urgent = (slack <= 0.0).astype(float)
    soft_slack_gate = 1.0 / (1.0 + np.exp(-7.772465461902413 * -norm_slack))
    uncertainty_mod = 1.0 + 0.8695147275111461 * norm_uncert * (ddl_urgent + (1.0 - ddl_urgent) * soft_slack_gate)
    protected_rank = norm_rank * (1.0 + 1.9138166133719576 * uncertainty_mod)
    successor_release_score = norm_work * norm_rank
    energy_uncert_penalty = norm_energy * np.where(norm_uncert > 0.7214211639109082, 1.0, 0.0)
    negative_slack_mask = ddl_urgent * 0.8695147275111461
    duration_risk_score = norm_duration * norm_uncert * negative_slack_mask
    wait_benefit = 1.0 - np.exp(-0.00012093913198998177 * (norm_wait + 7.068997472914438e-08))
    score = -ddl_urgent * 467.5230464148652 - protected_rank - successor_release_score - 1.1290821291292603 * norm_energy - wait_benefit + 0.0422316585310373 * duration_risk_score + 0.7463198100615581 * energy_uncert_penalty + 0.8190065717179699 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
