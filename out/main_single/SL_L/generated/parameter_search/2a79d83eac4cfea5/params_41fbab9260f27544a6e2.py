import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with exactly 12 parameters:
    - Removed redundant 'slack_pressure_tanh_scale' (merged into fixed logic using PARAMS["slack_decay_scale"]).
    - Uses power-law energy headroom gating with parametrized exponent.
    - Preserves work_density_bonus, critical_successor_boost, and robust normalization.
    - All numeric literals in body are in {-2,-1,0,1,2} or derived from np.finfo.
    """
    eps = 0.055299754105783895
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
        scale = 1.0094462365407095 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_pressure = np.tanh(-slack * 0.5593619980656627)
    slack_norm = 1.4455589505592505 * slack_pressure + 2.867563349083986 * (1.0 - np.tanh(slack * 0.5593619980656627))
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    fairness_headroom = np.divide(ready_wait_time, np.abs(slack) + eps)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_headroom_gate = np.exp(-np.power(np.clip(fairness_headroom, 0.0, None), 0.5100511425469892))
    energy_weight_adj = 1.858864474873329 * energy_headroom_gate
    rank_work_interaction = upward_rank * remaining_work
    rank_work_norm = mad_normalize(rank_work_interaction)
    median_uncertainty = np.median(uncertainty)
    ddl_protection_gate = ((slack >= 0.0) & (uncertainty <= median_uncertainty + eps)).astype(float)
    critical_successor_boost = 2.127063492664029 * rank_work_norm * ddl_protection_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 1.0820663913047415 * (1.0 - np.exp(-1.0820663913047415 * (wait_norm + eps)))
    unc_norm = mad_normalize(uncertainty)
    uncertainty_amplifier = 0.2532190079555544 * unc_norm * slack_pressure * ddl_protection_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_bonus = 0.27208660002846097 * work_density_norm * ddl_protection_gate
    score = slack_norm + 0.485053341301802 * duration_norm + energy_weight_adj * energy_norm - critical_successor_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
