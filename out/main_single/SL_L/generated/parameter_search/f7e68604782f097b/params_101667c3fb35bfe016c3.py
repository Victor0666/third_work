import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule integrating successor-release interaction and host-load conditional gating.
    
    Key structural mutations:
    - Replaces piecewise slack transformation with smooth tanh-scaled urgency + linear risk penalty
      for unified DDL pressure modeling (no discontinuities at zero)
    - Adds successor-release interaction: upward_rank * remaining_work, gated by slack feasibility
      to unblock critical paths early (counterfactual evidence: 'successor_release_interaction')
    - Introduces host-load conditional gate: energy term is suppressed only when both slack > 0 AND 
      uncertainty <= median_uncertainty, preventing premature energy optimization under risk
    - Replaces wait fairness with saturating exponential: 1 - exp(-gain * norm_wait), bounded and robust
    - Uses MAD-normalized features throughout; all operations epsilon-guarded and inf-safe
    """
    eps = 0.09998321130514982
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
        scale = 1.181066838228791 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    slack_urgency = np.tanh(slack * 0.451717260000786)
    slack_risk = 6.762988522406955 * slack_pressure
    slack_norm = slack_risk + 3.9975538232066676 * (1.0 - slack_urgency)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    median_unc = np.median(uncertainty)
    energy_suppression_gate = ((slack > 0.0) & (uncertainty <= median_unc + eps)).astype(float)
    energy_weight_adj = 1.8334485357456582 * (1.0 - energy_suppression_gate)
    rank_work_interaction = upward_rank * remaining_work
    rank_work_norm = mad_normalize(rank_work_interaction)
    successor_gate = (slack >= 0.0).astype(float)
    successor_boost = 1.5697417524700525 * rank_work_norm * successor_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 1.0042045816749017 * (1.0 - np.exp(-1.0042045816749017 * wait_norm))
    unc_norm = mad_normalize(uncertainty)
    uncertainty_amplifier = 0.9831026030679149 * unc_norm * slack_pressure
    score = slack_norm + 1.419302710618692 * duration_norm + energy_weight_adj * energy_norm - successor_boost + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
