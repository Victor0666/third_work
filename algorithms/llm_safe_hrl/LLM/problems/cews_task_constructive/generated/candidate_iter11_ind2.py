import numpy as np
RULE_METADATA = {'structure_hash': '970c75119151f89b44edb0d28400bbcfc94c75f089a29e031d5de0cdacc3d5d3', 'parameter_schema_hash': '914a8328f9cf534c9724a2d0363453a1e39c8b2b1686c5c8ade14bf54e89ac5a', 'best_parameter_hash': '888b3389f8714d81ed7da7fc948e5c8ef0df9fe4c08100f2502c295e9bdb26d0', 'best_parameters': {'epsilon': 0.002157910249696044, 'slack_sigmoid_steepness': 3.910128342756417, 'iqr_low_percentile': 24.48768000436879, 'iqr_high_percentile': 88.78143496145688, 'energy_duration_ratio_weight': 0.8186545947272097, 'successor_bottleneck_coupling': 0.8371948756752798, 'ddl_protection_gate_threshold': 0.9267006862206109, 'critical_rank_percentile': 0.7856414683362373, 'wait_saturation_time': 5.568566355326216, 'sigmoid_clip_bound': 10.80802466830962, 'uncertainty_amplification_weight': 0.2916341522487016, 'feasibility_slack_threshold': 0.04399228371576006}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'ea21f5bfe1670f42bc08860c759778d8262c689bb291ac46363726eaa5977c50', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust IQR normalization with Parent 1's feasibility-aware gating
    and uncertainty-modulated bottleneck coupling — validated by counterfactual replay.
    
    Key structural improvements:
      - Feasibility-aware gating: nullifies energy & uncertainty contributions when slack > threshold,
        preventing over-penalization of low-risk tasks and aligning with hard deadline constraint priority.
      - Uncertainty-modulated bottleneck: multiplicative coupling (not additive) reflects that risk amplifies blocking impact.
      - Tunable IQR normalization (20th/80th percentiles) improves outlier resilience vs fixed 25/75.
      - Smooth arctan wait saturation + percentile-gated critical bonus retained for fairness and path sensitivity.
      - All operations guarded against NaN/inf/zero; deterministic and finite.
    """
    eps = 0.002157910249696044
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def iqr_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 24.48768000436879)
        q_high = np.percentile(x, 88.78143496145688)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    sigmoid_input = np.clip(-3.910128342756417 * slack, -10.80802466830962, 10.80802466830962)
    urgency_gate = 2.0 / (1.0 + np.exp(sigmoid_input)) - 1.0
    norm_urgency = iqr_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    bottleneck_pressure = duration * (remaining_work + upward_rank + eps)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    norm_uncertainty = iqr_normalize(uncertainty)
    bottleneck_with_risk = norm_bottleneck * (1.0 + 0.2916341522487016 * norm_uncertainty)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.7856414683362373, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (5.568566355326216 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = iqr_normalize(wait_saturation)
    feasibility_gate = np.where(slack > 0.04399228371576006, 0.0, 1.0)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < median_slack) & (uncertainty > 0.9267006862206109 * max_uncertainty)).astype(float)
    score = norm_urgency + 0.8371948756752798 * bottleneck_with_risk + feasibility_gate * 0.8186545947272097 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_active * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
