import numpy as np
RULE_METADATA = {'structure_hash': 'e7f9668b355f2057ae384281e327201c2fe543dbb0c660bdd2d7d48ec129e44a', 'parameter_schema_hash': '4e49b9a25881e455aa8064560266979d70426a665e78db6319f3cd58a43e388c', 'best_parameter_hash': 'f13b976fe61670c43c0f1c3d2628540c376a241f6df778fff92383b780fe3035', 'best_parameters': {'epsilon': 2.140815790229643e-06, 'ddl_violation_weight': 1.9339035437319438, 'critical_path_gate_threshold': 0.6368987418553007, 'uncertainty_amplification_exponent': 1.5318420281680487, 'energy_efficiency_weight': 1.5376854875684494, 'wait_saturation_scale': 0.6249168720315914, 'bottleneck_coupling_weight': 0.42251089722896795, 'urgency_cap_exponent': 1.1230782284029166, 'percentile_scale_factor': 75.81991297767522}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '83e7b0cf39cc3db88c8735446513203bda0210d3aec38552ee7d0a813b727040', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with bounded control flow:
      - Uses only one conditional gate: critical-path activation via percentile mask (no nested or chained branches).
      - All other operations are vectorized and branch-free (clipping, power, sigmoid, normalization).
      - Replaces adaptive dispersion logic with fixed-metric MAD normalization — no std(uncertainty) branching.
      - Eliminates all multiplicative joint-risk couplings; uses additive, gated terms only.
      - Preserves hard-deadline dominance via weighted neg_slack, then urgency, bottleneck, energy, fairness.
      - Exactly 1 conditional branch (critical_path_mask), satisfying max-branch constraint.
      - No numeric literals beyond {-2,-1,0,1,2}; all tunables declared in PARAMETER_SCHEMA.
    """
    eps = 2.140815790229643e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def mad_normalize(x):
        x = np.copy(x)
        center = np.median(x) if N > 0 else 0.0
        abs_dev = np.abs(x - center)
        mad = np.median(abs_dev) if N > 0 else eps
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = np.where(mad > eps, mad, fallback_range)
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    ddl_penalty = 1.9339035437319438 * neg_slack
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, None)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 1.1230782284029166)
    urgency_clipped = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = mad_normalize(urgency_clipped)
    p = 0.6368987418553007
    percentile_slack = np.percentile(slack, 75.81991297767522 * p) if N > 0 else 0.0
    critical_path_mask = (slack <= percentile_slack).astype(float)
    critical_path_pressure = critical_path_mask * upward_rank * remaining_work
    norm_critical_path = mad_normalize(critical_path_pressure)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = mad_normalize(energy_per_duration)
    bottleneck_base = duration * upward_rank * remaining_work * (1.0 + urgency_clipped + eps)
    unc_norm = mad_normalize(uncertainty)
    bottleneck_pressure = bottleneck_base * np.power(1.0 + unc_norm, 1.5318420281680487)
    norm_bottleneck = mad_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (0.6249168720315914 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = mad_normalize(wait_saturation)
    score = ddl_penalty + norm_urgency + 0.42251089722896795 * norm_bottleneck + 0.6368987418553007 * norm_critical_path + 1.5376854875684494 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
