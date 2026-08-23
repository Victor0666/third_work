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
    eps = 0.0006885874295671563
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
        scale = 1.430466579407131 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    slack_urgency = np.tanh(slack * 0.6957434977116634)
    slack_risk = 11.406656098229341 * slack_pressure
    slack_norm = slack_risk + 2.7198428239853394 * (1.0 - slack_urgency)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    median_unc = np.median(uncertainty)
    energy_suppression_gate = ((slack > 0.0) & (uncertainty <= median_unc + eps)).astype(float)
    energy_weight_adj = 1.9353628066933366 * (1.0 - energy_suppression_gate)
    rank_work_interaction = upward_rank * remaining_work
    rank_work_norm = mad_normalize(rank_work_interaction)
    successor_gate = (slack >= 0.0).astype(float)
    successor_boost = 3.1532419307328907 * rank_work_norm * successor_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 0.40717089421118136 * (1.0 - np.exp(-0.40717089421118136 * wait_norm))
    unc_norm = mad_normalize(uncertainty)
    uncertainty_amplifier = 1.3827762966848884 * unc_norm * slack_pressure
    score = slack_norm + 0.05040782546462686 * duration_norm + energy_weight_adj * energy_norm - successor_boost + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
