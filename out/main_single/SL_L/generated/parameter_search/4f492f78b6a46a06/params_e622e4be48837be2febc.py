import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating counterfactual evidence:
    - Adds conditional DDL protection gate (slack >= 0 AND uncertainty <= median_uncertainty)
    - Replaces linear wait fairness with saturating exponential: 1 - exp(-gain * norm_wait)
    - Introduces successor-release interaction: upward_rank * remaining_work, gated by slack feasibility
    - Uses fairness headroom (wait_time / (|slack| + eps)) instead of raw slack to gate energy savings
    - Robustly normalizes all features using MAD scaled by robustness_mad_factor
    - All operations are epsilon-guarded and finite-value safe.
    """
    eps = 0.0006618551814995486
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.3595066024656424 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_pressure = np.tanh(-slack * 0.2577925170270794)
    slack_norm = 9.005728757733094 * slack_pressure + 3.9806765535153104 * (1.0 - np.tanh(slack * 0.19955915138477426))
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    fairness_headroom = np.divide(ready_wait_time, np.abs(slack) + eps)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_weight_adj = 1.9389669937455327 * np.clip(fairness_headroom, 0.0, 1.0)
    rank_work_interaction = upward_rank * remaining_work
    rank_work_norm = mad_normalize(rank_work_interaction)
    median_uncertainty = np.median(uncertainty)
    ddl_protection_gate = ((slack >= 0.0) & (uncertainty <= median_uncertainty + eps)).astype(float)
    critical_successor_boost = 1.0671669079999142 * rank_work_norm * ddl_protection_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 0.23245710498786953 * (1.0 - np.exp(-0.23245710498786953 * (wait_norm + eps)))
    unc_norm = mad_normalize(uncertainty)
    uncertainty_amplifier = 0.15288591117560785 * unc_norm * slack_pressure * ddl_protection_gate
    score = slack_norm + 0.6539116458992728 * duration_norm + energy_weight_adj * energy_norm - critical_successor_boost + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
