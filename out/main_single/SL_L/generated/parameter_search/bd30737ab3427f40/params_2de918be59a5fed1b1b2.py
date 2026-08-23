import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Self-evolved priority rule: replaces fragile median gating with smooth sigmoid criticality,
    removes noisy work-density bonus, introduces bounded work-slack coupling for deadline-constrained throughput,
    replaces volatile exponential wait boost with robust linear + uncertainty-offset scheme.
    All features use robust mean-abs normalization; no unbounded ops or loops.
    """
    eps = 1.1957867026262792e-07
    finfo = np.finfo(np.float64)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps)

    def robust_norm(x):
        denom = np.mean(np.abs(x)) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    duration_raw = min_exec_time + min_comm_time
    norm_duration = robust_norm(duration_raw)
    norm_rank = robust_norm(upward_rank)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    norm_work = robust_norm(remaining_work)
    slack_sign_mask = (slack < 0).astype(float)
    urgency_base = -norm_slack * slack_sign_mask
    urgency_clipped = np.clip(urgency_base, 0.0, 5.618121659197811)
    urgency_penalty = np.exp(3.3676868867436998 * urgency_clipped)
    critical_boost = 0.5696839980453534 * (1.0 + np.tanh(norm_rank / (0.5696839980453534 + eps)))
    slack_pressure = np.maximum(0.0, -norm_slack)
    uncertainty_coupled_pressure = 1.2498640209627734 * slack_pressure * norm_uncert
    wait_boost = 0.2910589421712608 * norm_wait + 0.3704371874685334 * (1.0 - norm_uncert)
    rank_energy_penalty = 0.9386176146480191 * norm_energy * norm_rank
    work_slack_bonus = 0.5507853691511009 * norm_work * np.maximum(0.0, -norm_slack)
    score = urgency_penalty + uncertainty_coupled_pressure + rank_energy_penalty + 0.9262451308637847 * norm_energy + 0.8131883960277463 * norm_duration - critical_boost - wait_boost - work_slack_bonus
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=-finfo.max)
    return score
