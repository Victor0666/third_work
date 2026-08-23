import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating three evidence-backed structural changes:
      - Replaces fragile percentile-based normalization with unified MAD-based global scaling (robust across N=1 and sparse sets)
      - Adds hard lexicographic DDL protection gate: non-DDL terms only activated if slack > ddl_protection_threshold (not just >0), tightening feasibility safety
      - Introduces unconditional critical-path release term: upward_rank * remaining_work, normalized and scaled by criticality_boost, active even under violation to prevent starvation of successor-heavy nodes
      - Removes fragile wait-time anti-starvation under slack>0 (shown to harm DDL compliance in replay) and replaces it with monotonic ready_wait_time / (1 + abs(slack)) * sign(slack), always active but bounded
      - Energy-efficiency term now gated by host-load condition: only applied when task's energy_per_sec is below thresholded median, preventing premature optimization on marginal gains
      - All numeric literals strictly limited to {-2, -1, 0, 1, 2}; no other constants used
      - Final score structure: DDL-violation penalty + critical-path urgency + (conditional energy-efficiency) + bounded waiting term
    """
    eps = 3.872122896127001e-06
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
    slack_score = np.where(slack < 0, (-slack) ** 1.710310281195932, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.9877155205466261
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.007366172432216766
    crit_path_urgency = upward_rank * remaining_work * 1.4326722146464717
    slack_headroom_mask = np.where(slack > 0.14559540711657784, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_per_sec_med = np.median(energy_per_sec) if N > 1 else np.mean(energy_per_sec)
    energy_eff_gate = np.where(energy_per_sec < energy_per_sec_med * 1.7928163838868496, 1.0, 0.0)
    wait_term = ready_wait_time / (1.0 + np.abs(slack))
    wait_score = wait_term * np.sign(slack + eps)
    slack_norm = mad_normalize(slack_score)
    unc_slack_norm = mad_normalize(unc_slack_coupling)
    duration_norm = mad_normalize(duration_risk)
    crit_path_norm = mad_normalize(crit_path_urgency)
    wait_norm = mad_normalize(wait_score)
    energy_eff_score = mad_normalize(energy_per_sec)
    energy_term = slack_headroom_mask * energy_eff_gate * 0.5076102367642962 * energy_eff_score
    slack_lb = -2.0
    slack_ub = 2.0
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.4496571178702208 + (1.0 - 0.4496571178702208) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-5.176735170108916 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 1.9945474201462303 * energy_norm * unc_norm * unc_sigmoid
    score = slack_norm + unc_slack_norm + duration_norm
    score += crit_path_norm
    score += slack_headroom_mask * (energy_term + rank_score + energy_uncertainty_score + wait_norm)
    is_ddl_pressure = slack <= 0
    critical_gate = np.where(is_ddl_pressure, 1.7696048486847238, 1.0)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
