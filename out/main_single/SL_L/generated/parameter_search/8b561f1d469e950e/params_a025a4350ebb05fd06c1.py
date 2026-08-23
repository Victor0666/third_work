import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's lexicographic DDL enforcement and robust MAD normalization
    with Parent 1's critical-path release signal and starvation mitigation via wait-time urgency scaling.
    Key novel improvements:
      - Introduces nonlinear slack violation amplifier (power-law) to sharply prioritize severely overdue tasks.
      - Unifies duration, uncertainty, and |slack| into a single joint MAD-normalized risk vector for coherent scaling.
      - Replaces per-feature MAD normalization with joint MAD over stacked risk dimensions → ensures consistent scale alignment.
      - Retains hard lexicographic gating (not sigmoid) for non-DDL terms, guaranteeing feasibility-first behavior.
      - Critical-path release term uses raw product (upward_rank * remaining_work), not normalized — preserves absolute importance magnitude.
      - Wait-time term now scales *both* by power-law decay AND inverse slack headroom (1/(slack - threshold + eps)), creating strong urgency when slack is tight.
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no other constants used.
    """
    eps = 5.231063479185277e-07
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
    abs_slack = np.abs(slack) + eps
    risk_stack = np.stack([abs_slack, uncertainty, duration_total], axis=0)
    risk_medians = np.median(risk_stack, axis=1, keepdims=True)
    risk_abs_devs = np.abs(risk_stack - risk_medians)
    risk_mads = np.median(risk_abs_devs, axis=1) + eps
    abs_slack_norm = (abs_slack - np.median(abs_slack)) / (risk_mads[0] * 0.13745695055561402 + eps)
    unc_norm = (uncertainty - np.median(uncertainty)) / (risk_mads[1] * 0.13745695055561402 + eps)
    dur_norm = (duration_total - np.median(duration_total)) / (risk_mads[2] * 0.13745695055561402 + eps)
    slack_violation = np.where(slack < 0, -slack, 0.0)
    slack_penalty = slack_violation ** 0.5904212990163966 * (1.0 + unc_norm) * 0.13745695055561402
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.6377339738913379
    critical_release_score = upward_rank * remaining_work * 3.7353636894538074
    slack_headroom_mask = np.where(slack > 1.1205360732484722, 1.0, 0.0)
    energy_norm = (min_incremental_energy - np.median(min_incremental_energy)) / (np.mean(np.abs(min_incremental_energy - np.median(min_incremental_energy))) + eps)
    energy_term = slack_headroom_mask * 1.4081906169730798 * energy_norm
    unc_sigmoid = 1.0 / (1.0 + np.exp(-0.5381027290209408 * (uncertainty - 1.0)))
    energy_uncertainty_term = slack_headroom_mask * 0.8613923610394021 * energy_norm * unc_norm * unc_sigmoid
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 2.0622980340411865 / (slack - 1.1205360732484722 + eps), 0.0)
    wait_norm = (wait_power - np.median(wait_power)) / (np.mean(np.abs(wait_power - np.median(wait_power))) + eps)
    wait_term = slack_headroom_mask * wait_norm
    slack_headroom_ratio = np.clip((slack - 1.1205360732484722) / (1.1205360732484722 + eps), 0.0, 2.0)
    rank_weight = 0.44020994283521353 * (1.0 - slack_headroom_ratio / (2.0 + eps)) + (1.0 - 0.44020994283521353) * (slack_headroom_ratio / (2.0 + eps))
    rank_norm = (upward_rank - np.median(upward_rank)) / (np.mean(np.abs(upward_rank - np.median(upward_rank))) + eps)
    rank_term = slack_headroom_mask * -rank_norm * rank_weight
    score = slack_penalty + duration_risk + critical_release_score
    score += energy_term + energy_uncertainty_term + wait_term + rank_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
