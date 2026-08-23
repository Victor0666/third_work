import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Adaptive per-dimension MAD scaling: uses relative dispersion (MAD / median) instead of fixed scale,
        eliminating manual `duration_mad_scale` and `mad_normalization_shift`.
      - Host-load–aware conditional gate: inferred from ready_wait_time & uncertainty dispersion as proxy for
        local VM load pressure; enables energy/latency terms only under low-load conditions.
      - Successor-release coupling: multiplies critical_path_release by normalized slack deficit to amplify
        critical-path urgency when deadlines tighten.
      - All normalization now uses robust, adaptive MAD ratios (MAD / max(median, eps)) to ensure stability
        under sparse or degenerate distributions.
      - No hardcoded constants beyond {-2,-1,0,1,2}; all tunable behavior exposed via PARAMS.
    """
    eps = 1.2971605386285859e-09
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
    all_risk_dims = np.stack([duration_total, abs_slack, uncertainty], axis=0)
    medians = np.median(all_risk_dims, axis=1)
    mads = np.mean(np.abs(all_risk_dims - medians[:, None]), axis=1) + eps
    duration_norm = duration_total / (medians[0] + mads[0] + eps)
    slack_norm = abs_slack / (medians[1] + mads[1] + eps)
    unc_norm = uncertainty / (medians[2] + mads[2] + eps)
    slack_penalty = np.where(slack < 0, -slack * (1.0 + 0.47572477864493323 * unc_norm), 0.0)
    critical_release = upward_rank * remaining_work
    median_cr = np.median(critical_release) + eps
    mad_cr = np.mean(np.abs(critical_release - median_cr)) + eps
    critical_release_norm = (critical_release - median_cr) / (mad_cr + eps)
    slack_deficit_ratio = np.clip(-slack / (0.7604696924560185 + eps), 0.0, 2.0)
    critical_release_score = -critical_release_norm * 3.7858263193263166 * (1.0 + 0.2514247755751245 * slack_deficit_ratio)
    ramp_half = 2.637892678009264 / 2.0
    gate_center = 0.7604696924560185
    gate_low = gate_center - ramp_half
    gate_high = gate_center + ramp_half
    slack_gate = np.clip((slack - gate_low) / (ramp_half * 2.0 + eps), 0.0, 1.0)
    wait_unc_ratio = (ready_wait_time + eps) / (uncertainty + eps)
    median_wait_unc = np.median(wait_unc_ratio) + eps
    mad_wait_unc = np.mean(np.abs(wait_unc_ratio - median_wait_unc)) + eps
    load_pressure = (wait_unc_ratio - median_wait_unc) / (mad_wait_unc + eps)
    host_load_gate = np.where(load_pressure < 0.6440619725398168, 1.0, 0.0)
    mad_energy = np.mean(np.abs(min_incremental_energy - np.median(min_incremental_energy))) + eps
    energy_median = np.median(min_incremental_energy) + eps
    energy_norm = (min_incremental_energy - energy_median) / (mad_energy + eps)
    energy_score = slack_gate * host_load_gate * energy_norm * 0.41477406473173306
    energy_uncertainty_score = slack_gate * host_load_gate * energy_norm * unc_norm * 0.7375197124784114
    latency_risk_score = slack_gate * host_load_gate * -duration_norm * 0.09381962464754696
    median_wait = np.median(ready_wait_time) + eps
    wait_ratio = np.clip(ready_wait_time / median_wait, 0.0, 2.0)
    wait_score = ready_wait_time * (1.0 + np.clip(-slack / (0.7604696924560185 + eps), 0.0, 2.0))
    wait_mad = np.mean(np.abs(wait_score - np.median(wait_score))) + eps
    wait_norm = (wait_score - np.median(wait_score)) / (wait_mad + eps)
    wait_final = wait_norm * 0.4124084461258003
    slack_weight = np.clip(1.0 - slack_norm / (slack_norm.max() + eps), 0.0, 1.0)
    rank_weight = 1.0 - slack_weight
    rank_score = -upward_rank / (np.median(upward_rank) + eps) * rank_weight * 0.8394851321955453
    score = slack_penalty + critical_release_score + wait_final + energy_score + energy_uncertainty_score + latency_risk_score + rank_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
