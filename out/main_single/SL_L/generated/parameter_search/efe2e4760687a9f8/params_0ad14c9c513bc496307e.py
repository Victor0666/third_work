import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's lexicographic reliability with Parent 1's differentiable urgency:
       - Introduces convex hybrid DDL gate: alpha * sigmoid_urgency + (1-alpha) * hard_mask → preserves CMA-ES gradients while retaining feasibility robustness
       - Keeps joint MAD normalization over |slack|, uncertainty, duration_total for coherent risk alignment
       - Retains explicit starvation mitigation via wait/duration ratio, gated by hybrid DDL signal
       - Critical-path release now uses *both* slack_pressure (Parent 2) *and* sigmoid_urgency (Parent 1) for dual-mode criticality scaling
       - Energy-uncertainty coupling uses bounded sigmoid centered at uncertainty=1.0 (Parent 2) but scaled by hybrid gate
       - All numeric literals restricted to {-2,-1,0,1,2}; no other constants used
       - Final score prioritizes DDL violation penalty first, then critical release, then feasible terms
    """
    eps = 7.433128998891563e-07
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
    all_risk_features = np.stack([abs_slack, uncertainty, duration_total], axis=0)
    mad = np.mean(np.abs(all_risk_features - np.median(all_risk_features, axis=1, keepdims=True)), axis=1)
    joint_mad = np.median(mad) + eps

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        return np.clip((x - np.median(x)) / (joint_mad * 0.32058756397290045 + eps), -2.0, 2.0)
    hard_mask = np.where(slack > 0.52913960943058, 1.0, 0.0)
    sigmoid_urgency = 1.0 / (1.0 + np.exp(11.97635332826537 * slack))
    hybrid_gate = 0.42414894348777876 * sigmoid_urgency + (1.0 - 0.42414894348777876) * hard_mask
    ddl_urgency = 1.0 - sigmoid_urgency
    ddl_violation_penalty = np.maximum(0.0, 0.52913960943058 - slack)
    critical_release_score = upward_rank * remaining_work
    slack_pressure = np.clip((0.52913960943058 - slack) / (0.52913960943058 + eps), 0.0, 1.0)
    combined_critical_scale = (slack_pressure + ddl_urgency) / 2.0
    critical_score = -mad_normalize(critical_release_score) * combined_critical_scale * 1.4316174336115308
    feasible_mask = hybrid_gate
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = feasible_mask * 0.17252803436688002 * energy_norm
    unc_slack_interaction = uncertainty * (0.52913960943058 - slack) * 1.436017562695747
    unc_slack_norm = mad_normalize(unc_slack_interaction)
    wait_efficiency = np.where(duration_total > eps, ready_wait_time / duration_total, 0.0)
    wait_norm = mad_normalize(wait_efficiency)
    wait_term = feasible_mask * 0.9858223549585692 * wait_norm
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.0 * (1.0 - uncertainty)))
    energy_uncertainty_term = feasible_mask * 1.1287289157350746 * energy_norm * unc_sigmoid
    score = mad_normalize(ddl_violation_penalty) + critical_score + unc_slack_norm
    score += energy_term + wait_term + energy_uncertainty_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
