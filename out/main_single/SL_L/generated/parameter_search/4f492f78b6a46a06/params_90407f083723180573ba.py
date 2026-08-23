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
    eps = 0.074714874338628
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
        scale = 1.4703714145732911 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_pressure = np.tanh(-slack * 0.11768377235221962)
    slack_norm = 6.60382866655201 * slack_pressure + 2.1456906536163167 * (1.0 - np.tanh(slack * 0.43398862316875325))
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    fairness_headroom = np.divide(ready_wait_time, np.abs(slack) + eps)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_weight_adj = 1.5057537525731701 * np.clip(fairness_headroom, 0.0, 1.0)
    rank_work_interaction = upward_rank * remaining_work
    rank_work_norm = mad_normalize(rank_work_interaction)
    median_uncertainty = np.median(uncertainty)
    ddl_protection_gate = ((slack >= 0.0) & (uncertainty <= median_uncertainty + eps)).astype(float)
    critical_successor_boost = 0.1010229166822569 * rank_work_norm * ddl_protection_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 0.31700383780722485 * (1.0 - np.exp(-0.31700383780722485 * (wait_norm + eps)))
    unc_norm = mad_normalize(uncertainty)
    uncertainty_amplifier = 1.6346367000383581 * unc_norm * slack_pressure * ddl_protection_gate
    score = slack_norm + 0.6928714323046845 * duration_norm + energy_weight_adj * energy_norm - critical_successor_boost + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
