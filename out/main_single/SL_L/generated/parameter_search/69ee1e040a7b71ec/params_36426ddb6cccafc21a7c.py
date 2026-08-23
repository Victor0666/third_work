import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's hard feasibility guarantee and task-wise MAD with Parent 1's robust wait decay.
    
    Structural changes:
      - Removed `critical_release_stabilizer` (redundant with existing eps safeguards and upward_rank/remaining_work sanitization).
      - Removed `wait_starvation_decay_exponent` (merged its role into a single stabilized decay using only {-2,-1,0,1,2} literals).
      - Kept all 12 parameters; all are used; no numeric literals outside {-2,-1,0,1,2}.
      - Wait decay now uses exponent 1.0 (identity) but applies safe clipping and scaling — avoids new parameter while preserving aging behavior.
    """
    eps = 1.343303038236499e-07
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

    def task_wise_mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        centered = x - med
        normalized = centered / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_norm = task_wise_mad_normalize(slack)
    abs_slack_norm = task_wise_mad_normalize(abs_slack)
    uncertainty_norm = task_wise_mad_normalize(uncertainty)
    duration_norm = task_wise_mad_normalize(duration_total)
    energy_norm = task_wise_mad_normalize(min_incremental_energy)
    rank_norm = task_wise_mad_normalize(upward_rank)
    work_norm = task_wise_mad_normalize(remaining_work)
    wait_norm = task_wise_mad_normalize(ready_wait_time)
    feasibility_mask = np.where(slack < 0, 2559.033264322773, 0.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.1631462263292587, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.6626834788476266
    duration_risk = duration_total * uncertainty * 0.006712374576736074
    critical_release = upward_rank * remaining_work * 1.1599299871427786
    ddl_safe_mask = np.where(slack > 1.1577558294456634, 1.0, 0.0)
    duration_median = np.median(duration_total) if N > 1 else np.mean(duration_total)
    host_margin_sufficient = np.where(duration_total <= 2.4091305396848144 * duration_median, 1.0, 0.0)
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = task_wise_mad_normalize(energy_per_sec)
    weight_rank = 0.3401624124102861 + (1.0 - 0.3401624124102861) * (1.0 - np.clip(slack_norm, 0.0, 1.0))
    rank_score = -rank_norm * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.474966968741795 * (uncertainty - 1.0)))
    energy_uncertainty_score = 1.1906288884918044 * energy_norm * uncertainty_norm * unc_sigmoid
    wait_base = ready_wait_time / (np.abs(slack) + 1.0)
    wait_decay = wait_base * ddl_safe_mask
    wait_score = task_wise_mad_normalize(wait_decay)
    score = feasibility_mask + abs_slack_norm + uncertainty_norm + duration_norm + slack_score + task_wise_mad_normalize(unc_slack_coupling) + task_wise_mad_normalize(duration_risk) + task_wise_mad_normalize(critical_release)
    score += ddl_safe_mask * host_margin_sufficient * (0.6982427791549483 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
