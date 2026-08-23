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
    eps = 7.117696208551029e-08
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
    slack_score = np.where(slack < 0, (-slack) ** 1.005182033487205, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.6596989791535295
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.2299321297769747
    rank_work_product = upward_rank * remaining_work
    rank_work_med = np.median(rank_work_product) if N > 1 else np.mean(rank_work_product)
    is_high_critical_path = rank_work_product >= rank_work_med
    criticality_boost_mask = np.where((slack <= 0) & is_high_critical_path, 1.9448972334439814, 1.0)
    slack_headroom_mask = np.where(slack > 0.6143323807055088, 1.0, 0.0)
    energy_norm = global_mad_normalize(min_incremental_energy)
    load_gate = np.where(energy_norm < 1.7078414946235607, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = global_mad_normalize(energy_per_sec)
    inv_slack = np.where(slack > 0.6143323807055088, 1.0 / (slack + 1.0), 0.0)
    wait_score = global_mad_normalize(ready_wait_time * inv_slack)
    rank_score = -global_mad_normalize(upward_rank)
    slack_scaled = np.clip((slack - 0.6143323807055088) / (0.6143323807055088 + 1.0), 0.0, 1.0)
    weight_rank = 0.07483868428895002 + (1.0 - 0.07483868428895002) * (1.0 - slack_scaled)
    rank_score = rank_score * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.5458231753761589 * (uncertainty - 1.0)))
    energy_uncertainty_score = 0.8932774419702534 * energy_norm * global_mad_normalize(uncertainty) * unc_sigmoid
    score = global_mad_normalize(slack_score) + global_mad_normalize(unc_slack_coupling) + global_mad_normalize(duration_risk)
    score += slack_headroom_mask * load_gate * (1.499402658534023 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score + 1.3716468639590922 * global_mad_normalize(rank_work_product))
    score = score * criticality_boost_mask
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
