import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Novel priority rule with three-mode logic: critical (negative slack), urgent (tight slack), and normal.
    
    Key innovations:
    - Critical mode: triggers hard penalty for slack < critical_slack_threshold, scaled by slack_penalty_factor
      and amplified by uncertainty_slack_coupling → ensures immediate attention to at-risk tasks.
    - Urgent mode: applies rank_leverage to upward_rank only when slack is small but non-negative, 
      balancing critical-path progress and deadline headroom.
    - Normal mode: emphasizes energy efficiency via nonlinear_energy_exponent (superlinear penalty) and 
      bounded fairness via wait_fairness_gain.
    - All features are robustly normalized using mean-abs + epsilon; no unbounded ops.
    - No division by raw inputs; all denominators guarded.
    - Returns strictly (N,) array of finite floats.
    """
    eps = 4.912214696577536e-05
    nan_sub = 6.9663327425181285e-06
    inf_clip = 1156557.6269941283
    N = len(slack)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=nan_sub, posinf=inf_clip, neginf=-inf_clip)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=nan_sub, posinf=inf_clip, neginf=-inf_clip)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=nan_sub, posinf=inf_clip, neginf=-inf_clip)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=inf_clip, neginf=-inf_clip)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=nan_sub, posinf=inf_clip, neginf=-inf_clip)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=nan_sub, posinf=inf_clip, neginf=-inf_clip)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=inf_clip, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=nan_sub, posinf=inf_clip, neginf=-inf_clip)

    def robust_norm(x):
        x_abs = np.abs(x)
        denom = np.mean(x_abs) + eps
        return x / denom
    norm_exec_comm = robust_norm(min_exec_time + min_comm_time)
    norm_energy = robust_norm(min_incremental_energy)
    norm_rank = robust_norm(upward_rank)
    norm_work = robust_norm(remaining_work)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    slack_pressure = robust_norm(slack)
    gate_width = 0.025402608781944298
    is_critical = np.clip((0.13308396489469665 - slack) / (eps + gate_width), 0.0, 1.0)
    slack_risk_amplifier = np.maximum(0.0, -slack_pressure) * (1.0 + 0.03531233594839295 * norm_uncert)
    critical_penalty = 0.5240442198472144 * is_critical * slack_risk_amplifier
    is_urgent = np.where((slack >= 0.0) & (slack < 0.13308396489469665), 1.0, 0.0)
    urgent_rank_boost = 3.07686355709847 * is_urgent * norm_rank
    energy_penalty = 1.2470610515212195 * np.abs(norm_energy) ** 1.433494799295036
    fairness_bonus = -0.32727360300565755 * norm_wait
    duration_urgency = 1.0108812549892812 * norm_exec_comm * (1.0 - is_critical)
    score = critical_penalty + energy_penalty + duration_urgency - urgent_rank_boost + fairness_bonus
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected (N,) but got {score.shape}'
    return score
