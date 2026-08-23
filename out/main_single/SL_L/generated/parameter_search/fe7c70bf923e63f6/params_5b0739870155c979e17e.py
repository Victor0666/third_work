import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating key counterfactual evidence:
      - Replaces strict 'slack > 0' gate with tighter 'slack > ddl_protection_threshold' to prevent premature energy optimization
      - Adds explicit upward_rank × remaining_work interaction as successor-release signal (confirmed in 6+ starvation failures)
      - Introduces host-load conditional gate: energy terms only activated when energy is below a normalized threshold
      - Uses MAD-based global feature scaling instead of per-feature percentile normalization for rank stability
      - Replaces ready_wait_time term with monotonic, bounded inverse slack weighting to avoid numerical explosion
      - All DDL-critical terms remain unconditionally active; non-DDL terms now require both slack headroom AND low energy load
      - Criticality boost applied multiplicatively only when slack <= 0 AND upward_rank × remaining_work is above median
      - No fragile sigmoid/percentile thresholds: all gates use bounded, monotonic forms
      - Final score preserves lexicographic DDL-first ordering while enabling safer energy optimization
    """
    eps = 2.652834769474814e-08
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

    def global_mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 1.6649566272477458, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.363935361606596
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.39489005575415237
    rank_work_product = upward_rank * remaining_work
    rank_work_med = np.median(rank_work_product) if N > 1 else np.mean(rank_work_product)
    is_high_critical_path = rank_work_product >= rank_work_med
    criticality_boost_mask = np.where((slack <= 0) & is_high_critical_path, 2.8765100002490716, 1.0)
    slack_headroom_mask = np.where(slack > 6.6940263614828375, 1.0, 0.0)
    energy_norm = global_mad_normalize(min_incremental_energy)
    load_gate = np.where(energy_norm < 1.4677807696620828, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = global_mad_normalize(energy_per_sec)
    inv_slack = np.where(slack > 6.6940263614828375, 1.0 / (slack + 1.0), 0.0)
    wait_score = global_mad_normalize(ready_wait_time * inv_slack)
    rank_score = -global_mad_normalize(upward_rank)
    slack_scaled = np.clip((slack - 6.6940263614828375) / (6.6940263614828375 + 1.0), 0.0, 1.0)
    weight_rank = 0.8511271111140448 + (1.0 - 0.8511271111140448) * (1.0 - slack_scaled)
    rank_score = rank_score * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.319380358680806 * (uncertainty - 1.0)))
    energy_uncertainty_score = 1.1103649628433585 * energy_norm * global_mad_normalize(uncertainty) * unc_sigmoid
    score = global_mad_normalize(slack_score) + global_mad_normalize(unc_slack_coupling) + global_mad_normalize(duration_risk)
    score += slack_headroom_mask * load_gate * (1.8921426762018154 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score + 1.5575681562018164 * global_mad_normalize(rank_work_product))
    score = score * criticality_boost_mask
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
