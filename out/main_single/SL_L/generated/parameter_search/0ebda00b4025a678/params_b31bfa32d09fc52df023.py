import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Novel priority rule: prioritizes deadline-criticality first via robust slack gating,
    couples uncertainty with slack to amplify risk awareness, uses exponential urgency for negative slack,
    applies conditional criticality boost, and penalizes high-energy + high-rank combinations.
    Anti-starvation uses soft exponential wait boost instead of linear scaling.
    All features are robustly normalized using mean-abs + epsilon; no unbounded ops.
    """
    eps = 1.264697690033951e-06

    def robust_norm(x):
        x = np.asarray(x, dtype=np.float64)
        denom = np.mean(np.abs(x)) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    norm_duration = robust_norm(min_exec_time + min_comm_time)
    norm_rank = robust_norm(upward_rank)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    slack_sign_mask = (slack < 0).astype(float)
    urgency_base = -norm_slack * slack_sign_mask
    urgency_clipped = np.clip(urgency_base, 0.0, 5.588950223933505)
    urgency_penalty = np.exp(2.5000633950857347 * urgency_clipped)
    rank_median = np.median(norm_rank)
    critical_mask = (norm_rank >= rank_median).astype(float)
    critical_boost = 2.6154818290332744 * norm_rank * critical_mask
    slack_pressure = np.maximum(0.0, -norm_slack)
    uncertainty_coupled_pressure = 0.6334567974861922 * slack_pressure * norm_uncert
    wait_boost = 1.0 - np.exp(-0.31110522368492355 * norm_wait)
    rank_energy_penalty = 0.9547708406809117 * norm_energy * norm_rank
    score = urgency_penalty + uncertainty_coupled_pressure + rank_energy_penalty + 0.16472721613302144 * norm_energy + 0.06246366446707146 * norm_duration - critical_boost - wait_boost
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=-finfo.max)
    return score
