import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's robust MAD normalization and sigmoid gating with Parent 1's raw slack fidelity.
       Introduces explicit rank-slack coupling and removes redundant slack_normalization_mode to stay within 12-parameter limit.
       All operations safeguarded against NaN/inf; deterministic and shape-compliant."""
    eps = 0.0009854231010225972
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
        x_abs = np.abs(x)
        center = np.median(x_abs) if N > 1 else x_abs[0]
        spread = np.median(np.abs(x_abs - center)) if N > 1 else np.abs(x_abs[0] - center) + eps
        return (x_abs - center) / (spread + eps)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.9670187124668845
    slack_pressure = np.clip(-slack, 0.0, None)
    slack_pressure_norm = (slack_pressure - np.min(slack_pressure)) / (np.max(slack_pressure) - np.min(slack_pressure) + eps) if N > 1 else np.zeros_like(slack_pressure)
    slack_pressure_norm = np.clip(slack_pressure_norm, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-5.591391580201016 * (slack_pressure_norm - 1.0)))
    coupled_rank_boost = 0.9994460497753886 * slack_pressure_norm
    boosted_rank = norm_rank * (1.0 + 1.4133983537811967 * rank_gate + coupled_rank_boost)
    uncert_gate = np.where(norm_uncert > 0.9966205342564868, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure_norm
    wait_benefit = 1.0 - np.exp(-0.025268943422683432 * (norm_wait + 4.271640141985826e-06))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +raw_slack_penalty - boosted_rank - 1.0157756089057537 * norm_energy - norm_duration - wait_benefit + 0.5214325179992519 * duration_risk_score + 0.442218409516727 * energy_uncert_penalty + 0.44811709476365635 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
