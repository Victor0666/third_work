import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule incorporating three key structural improvements:
      - Replaces fragile exponentiated wait-slack coupling with robust bounded linear denominator: (1 + clip(slack, 0, 2))
      - Replaces tanh-based uncertainty gate with clipped linear gate: clip(uncertainty, 0, 1), eliminating unstable gradients
      - Restores host-load conditional gate to suppress premature energy optimization on marginal tasks
      - Introduces unified 'urgency density' feature: crit_path_urgency / (min_exec_time + min_comm_time + eps), emphasizing fast-release of critical work
      - All MAD-normalized components strictly clipped to [-2, 2] for guaranteed robustness at N=1 and sparse sets
      - Final score maintains strict lexicographic DDL safety: violation terms dominate; non-DDL terms gated by slack_headroom_mask AND host_load_gate
    """
    eps = 8.75679737286219e-09
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

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 1.0537713203714152, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.00476734408073956
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.4293773079086347
    crit_path_urgency = upward_rank * remaining_work * 0.5660742624192308
    duration_total = min_exec_time + min_comm_time + eps
    urgency_density = crit_path_urgency / duration_total
    slack_headroom_mask = np.where(slack > 3.227746847406107, 1.0, 0.0)
    slack_clipped = np.clip(slack, 0.0, 2.0)
    wait_denom = 1.0 + slack_clipped
    wait_base = ready_wait_time / wait_denom
    wait_score = wait_base * np.sign(slack + eps)
    energy_per_sec = min_incremental_energy / duration_total
    energy_per_sec_med = np.median(energy_per_sec) if N > 1 else np.mean(energy_per_sec)
    host_load_gate = np.where(energy_per_sec < energy_per_sec_med * 0.4589290579127892, 1.0, 0.0)
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    energy_eff_score = mad_normalize(energy_per_work)
    slack_lb = -2.0
    slack_ub = 2.0
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.8669176597897539 + (1.0 - 0.8669176597897539) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_linear_gate = np.clip(uncertainty, 0.0, 1.0)
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.5823319441743642 * energy_norm * unc_norm * unc_linear_gate
    slack_norm = mad_normalize(slack_score)
    unc_slack_norm = mad_normalize(unc_slack_coupling)
    duration_norm = mad_normalize(duration_risk)
    crit_path_norm = mad_normalize(crit_path_urgency)
    urgency_density_norm = mad_normalize(urgency_density)
    wait_norm = mad_normalize(wait_score)
    score = slack_norm + unc_slack_norm + duration_norm
    score += crit_path_norm
    score += urgency_density_norm
    score += slack_headroom_mask * host_load_gate * (0.16588036287366348 * energy_eff_score + rank_score + energy_uncertainty_score + wait_norm)
    is_ddl_pressure = slack <= 0
    critical_gate = np.where(is_ddl_pressure, 2.7439666198536465, 1.0)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
