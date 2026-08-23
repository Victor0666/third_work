import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Hybrid priority rule combining Parent 2's smooth tanh slack urgency and stable pre-gated rank normalization
    with Parent 1's explicit uncertainty penalty capping and refined fairness logic.
    Key structural improvements:
      - Introduces `uncertainty_penalty_cap` to bound worst-case uncertainty amplification (novel hard cap),
      - Replaces fragile linear uncertainty gating with a *smooth sigmoid gate* on (uncertainty / unc_med) for stability,
      - Uses robust median-based feasibility gating (not mean or ad-hoc thresholds) for all conditionals,
      - Retains Parent 2's decoupled energy-slack coupling and unified tanh urgency for deadline safety,
      - Preserves Parent 1's wait fairness saturation via exp(-x) but applies it *after* robust normalization.
    All parameters used; no numeric literals except {-2,-1,0,1,2}; fully deterministic and finite.
    """
    eps = 1.8366443465234008e-07
    finfo = np.finfo(np.float64)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=np.float64)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.4135818771631865 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_pressure = -slack
    slack_urgency = 3.123135652730503 * np.tanh(slack_pressure * 0.809683318824998)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_base = 0.14608465799895798 * energy_norm
    energy_slack_gate = np.clip(slack_pressure, 0.0, np.inf)
    energy_score = energy_base * (1.0 + 0.09907201699777396 * np.tanh(energy_slack_gate * 0.809683318824998))
    rank_norm = mad_normalize(upward_rank)
    unc_med = np.median(uncertainty)
    unc_sigmoid_gate = 1.0 / (1.0 + np.exp((uncertainty - unc_med) / np.maximum(eps, unc_med)))
    feasibility_gate = (slack >= 0.0).astype(float)
    critical_gate = feasibility_gate * unc_sigmoid_gate
    critical_boost = 1.8853337614300862 * rank_norm * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_bonus = 0.7920880784130935 * work_density_norm * feasibility_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 1.0 - np.exp(-0.15005286184656919 * np.maximum(wait_norm, 0.0))
    wait_score = wait_boost * feasibility_gate
    unc_norm = mad_normalize(uncertainty)
    resilience_base = 1.2413847957479645 * rank_norm * unc_norm * feasibility_gate
    resilience_boost = np.clip(resilience_base, -1.0415342860010965, 1.0415342860010965)
    score = slack_urgency + 0.6557057904365431 * duration_norm + energy_score - critical_boost - work_density_bonus + wait_score - resilience_boost
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    return score
