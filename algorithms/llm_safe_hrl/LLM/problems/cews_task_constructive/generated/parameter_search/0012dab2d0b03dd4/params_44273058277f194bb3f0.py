import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with slack-aware uncertainty scaling and conditional critical-path suppression:
      - Replaces brittle percentile gating with smooth, differentiable slack-feasibility modulation:
        urgency and energy terms scaled by sigmoid((median_slack - slack)^p / (std_slack + eps)) * uncertainty^q.
      - Critical-path pressure is *suppressed* when slack > threshold (in std units), enforcing hard DDL safety first.
      - Feasibility-energy tradeoff uses (1 - sigmoid(slack_feasibility)) to smoothly shift from deadline compliance to energy minimization.
      - Fairness term uses exponential decay (not inverted sigmoid) for tighter, bounded anti-starvation control.
      - All normalizations use epsilon-guarded min-max; no std-only fallbacks — robust for N=1.
      - Final composition: urgency > feasibility-energy > conditional critical-path > fairness.
    """
    eps = 1.0000424764261244e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    finfo = np.finfo(float)
    median_slack = np.median(slack) if N > 0 else 0.0
    std_slack = np.std(slack) if N > 1 else eps
    slack_feasibility_raw = (slack - median_slack) / (std_slack + eps)
    slack_feasibility = np.clip(slack_feasibility_raw, -2.0, 2.0)
    slack_sigmoid = 1.0 / (1.0 + np.exp(-slack_feasibility))
    urgency_base = (median_slack - slack) / (std_slack + eps)
    urgency_powered = np.power(np.abs(urgency_base) + eps, 0.6243343134173862)
    urgency_sign = np.sign(urgency_base)
    urgency_scaled = urgency_sign * urgency_powered
    uncertainty_coupled = np.power(uncertainty + eps, 1.9798300225389238)
    raw_urgency = 1.0 / (1.0 + np.exp(-urgency_scaled / (uncertainty_coupled + eps)))
    u_min, u_max = (np.min(raw_urgency), np.max(raw_urgency))
    u_range = np.maximum(u_max - u_min, eps)
    norm_urgency = (raw_urgency - u_min) / (u_range + eps)
    energy_weight = 1.0 - slack_sigmoid
    energy_term = min_incremental_energy * energy_weight * uncertainty_coupled
    e_min, e_max = (np.min(energy_term), np.max(energy_term))
    e_range = np.maximum(e_max - e_min, eps)
    norm_energy = (energy_term - e_min) / (e_range + eps)
    cp_active_mask = slack_feasibility_raw <= 0.10903871045132485
    critical_path_pressure = np.where(cp_active_mask, upward_rank * remaining_work, 0.0)
    cp_min, cp_max = (np.min(critical_path_pressure), np.max(critical_path_pressure))
    cp_range = np.maximum(cp_max - cp_min, eps)
    norm_critical_path = np.where(cp_active_mask, (critical_path_pressure - cp_min) / (cp_range + eps), 0.0)
    w_min, w_max = (np.min(ready_wait_time), np.max(ready_wait_time))
    w_range = np.maximum(w_max - w_min, eps)
    norm_wait = (ready_wait_time - w_min) / (w_range + eps)
    fairness_penalty = np.exp(-0.32192679875565194 * norm_wait)
    score = norm_urgency + 0.8438765156984356 * norm_energy + norm_critical_path - fairness_penalty
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
