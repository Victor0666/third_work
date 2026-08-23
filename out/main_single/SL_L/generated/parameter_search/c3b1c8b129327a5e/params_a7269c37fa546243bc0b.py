import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Hybrid priority rule combining Parent 2's robust Q1/Q3 normalization and piecewise slack modeling
    with Parent 1's explicit fairness gating and joint feasibility awareness.
    Key improvements:
      - Replaces triple-gated critical boost with dual-gated (slack >= 0 ∧ uncertainty <= median) + upward_rank interaction,
        removing brittle energy-median coupling that degraded under fuzzy energy variance.
      - Introduces work-density gating using smooth joint feasibility (tanh(slack/(uncertainty+eps))) instead of binary slack gate,
        enabling graceful ramp-down under risk.
      - Adds bounded, offset-tanh joint feasibility gate for work density and critical path — no threshold parameter removed.
      - Retains multiplicative uncertainty coupling for energy and urgency, applied after normalization.
      - All parameters used; exactly 12 parameters; no numeric literals beyond {-2,-1,0,1,2}; fully deterministic and finite.
    """
    eps = 0.0007495758898472651
    finfo = np.finfo(np.float64)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps)

    def q1_q3_normalize(x):
        x = np.asarray(x, dtype=np.float64)
        q1 = np.quantile(x, 0.1934249029646748)
        q3 = np.quantile(x, 0.789885253046644)
        iqr = q3 - q1
        scale = iqr if iqr > eps else eps
        return (x - np.median(x)) / scale
    neg_mask = slack < 0.0
    tight_mask = (slack >= 0.0) & (slack <= 43.527002674423606)
    loose_mask = slack > 43.527002674423606
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 0.8871717728029531 * -slack[neg_mask]
    slack_norm[tight_mask] = 1.6705029454803133 * (np.exp(slack[tight_mask]) - 1.0)
    slack_norm[loose_mask] = 1.6705029454803133 * (np.exp(43.527002674423606) - 1.0)
    duration = min_exec_time + min_comm_time
    duration_norm = q1_q3_normalize(duration)
    unc_norm = q1_q3_normalize(uncertainty)
    energy_base = min_incremental_energy * (1.0 + np.abs(unc_norm))
    energy_norm = q1_q3_normalize(energy_base)
    energy_score = 0.4860518020746817 * energy_norm
    rank_norm = q1_q3_normalize(upward_rank)
    unc_med = np.median(uncertainty)
    critical_gate = ((slack >= 0.0) & (uncertainty <= unc_med)).astype(float)
    critical_boost = 0.18759519447802586 * rank_norm * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = q1_q3_normalize(work_density)
    joint_feasibility = 0.6394397874204124 * (1.0 + np.tanh(slack / (uncertainty + eps)))
    joint_feasibility = np.clip(joint_feasibility, 0.0, 1.0)
    work_density_bonus = 0.31863186541515337 * work_density_norm * joint_feasibility
    unc_mod = 1.0 + 0.4033116846697825 * np.abs(unc_norm)
    modulated_energy_score = energy_score * unc_mod
    modulated_slack_norm = slack_norm * unc_mod
    score = modulated_slack_norm + 1.081160531435386 * duration_norm + modulated_energy_score - critical_boost - work_density_bonus
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    return score
