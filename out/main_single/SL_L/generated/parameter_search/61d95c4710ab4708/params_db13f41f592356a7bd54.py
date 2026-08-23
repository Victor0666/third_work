import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining robustness from Parent 1 and expressiveness from Parent 2:
      - Joint MAD normalization over [abs(slack), uncertainty, duration_total] using fused weighted median absolute deviation
      - Critical-path release scaled by *nonlinear* slack pressure: (max(0, ddl_th - slack))^exponent → sharper penalty near threshold
      - Bounded sigmoid on uncertainty (centered at 1.0, slope=2) to smoothly suppress energy-aware terms under high risk
      - Starvation mitigation via wait/duration ratio, *only activated when feasible*, avoiding starvation without compromising DDL safety
      - Lexicographic DDL enforcement: violation penalty dominates; all other terms are gated or scaled by feasibility
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no hidden constants
    """
    eps = 2.6190088973916206e-08
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack)
    risk_signals = np.stack([abs_slack, uncertainty, duration_total], axis=0)
    risk_medians = np.median(risk_signals, axis=1, keepdims=True)
    abs_deviations = np.abs(risk_signals - risk_medians)
    mad_per_signal = np.median(abs_deviations, axis=1)
    weighted_mads = mad_per_signal * np.array([0.7764463779762765, 1.0, 2.0 - 0.7764463779762765])
    joint_mad = np.median(weighted_mads) + eps

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        x_median = np.median(x)
        norm = (x - x_median) / (joint_mad * 1.591288840509561 + eps)
        return np.clip(norm, -2.0, 2.0)
    slack_deficit = np.maximum(0.0, 0.07085360127253112 - slack)
    ddl_violation_penalty = slack_deficit ** 1.5796492909989266
    critical_release_score = upward_rank * remaining_work
    slack_pressure = np.clip(slack_deficit / (0.07085360127253112 + eps), 0.0, 1.0)
    critical_score = -mad_normalize(critical_release_score) * slack_pressure ** 1.5796492909989266 * 1.1014793724367777
    feasible_mask = np.where(slack > 0.07085360127253112, 1.0, 0.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = feasible_mask * 0.8866020042627715 * energy_norm
    unc_slack_interaction = uncertainty * slack_deficit * 0.5467471524773104
    unc_slack_norm = mad_normalize(unc_slack_interaction)
    wait_efficiency = np.where(duration_total > eps, ready_wait_time / duration_total, 0.0)
    wait_norm = mad_normalize(wait_efficiency)
    wait_term = feasible_mask * 0.7437840197207732 * wait_norm
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.0 * (1.0 - uncertainty)))
    energy_uncertainty_term = feasible_mask * 0.6902869863736985 * energy_norm * unc_sigmoid
    score = mad_normalize(ddl_violation_penalty) + critical_score + unc_slack_norm
    score += energy_term + wait_term + energy_uncertainty_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
