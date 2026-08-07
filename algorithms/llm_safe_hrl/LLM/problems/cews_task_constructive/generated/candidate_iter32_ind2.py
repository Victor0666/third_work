import numpy as np
RULE_METADATA = {'structure_hash': '75c6b606baf611bde474ca075995a644112f2069bba82458013e869141e905de', 'parameter_schema_hash': 'c7eff470c5f4ea6dce493079c9108c7cf170ff4627b996e46b27697c7b9d569e', 'best_parameter_hash': '649acd22067892dcff3617c4ce9a911087a84c5e770512face49492d9f4d4c98', 'best_parameters': {'epsilon': 1.8941038532088433e-06, 'ddl_protection_threshold': 0.03400453209684031, 'bottleneck_uncertainty_amplification': 1.7818449610465628, 'upward_rank_remaining_work_weight': 0.5026338047530892, 'successor_release_coupling': 0.09442926164015969, 'wait_saturation_clipping_factor': 0.6594014585310611, 'energy_priority_weight': 1.2550763772325206}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '332116fb919d2ea31c9ee100fe2ff89c28c9bb16a760b0d8c18a27a2603a7131', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule synthesizing strengths from both parents:
      - Retains Parent 2's robust successor-release coupling: (1 + slack/median_slack) * (1 + uncertainty)
      - Adopts Parent 1's explicit DDL-protection activation condition using normalized uncertainty
      - Introduces novel energy-aware prioritization: min_incremental_energy is weighted *only* when slack is non-negative,
        avoiding energy optimization at the cost of deadline violation.
      - Unifies normalization via ddl_aware_normalize with outlier-resistant clipping.
      - Adds conditional energy term: active only when slack >= 0 (feasible to optimize energy without violating DDL).
      - Preserves anti-starvation via bounded sigmoid on ready_wait_time, scaled by clipping factor.
      - All operations are finite, deterministic, epsilon-guarded, and avoid NaN/inf/zero.
    """
    eps = 1.8941038532088433e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    unc_min, unc_max = (np.min(uncertainty), np.max(uncertainty))
    unc_range = np.maximum(unc_max - unc_min, eps)
    unc_normalized = (uncertainty - unc_min) / (unc_range + eps)
    ddl_protection_active = (slack <= median_slack) & (unc_normalized > 0.03400453209684031)
    critical_pressure = upward_rank * remaining_work
    slack_ratio = np.where(np.abs(median_slack) > eps, slack / (np.abs(median_slack) + eps), 0.0)
    slack_ratio = np.clip(slack_ratio, -2.0, 2.0)
    successor_release = (1.0 + slack_ratio) * (1.0 + uncertainty)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    bottleneck_base = duration * critical_pressure * successor_release
    bottleneck_pressure = np.where(ddl_protection_active, bottleneck_base * np.power(1.0 + unc_normalized, 1.7818449610465628), bottleneck_base)
    energy_term = np.where(slack >= 0.0, min_incremental_energy, 0.0)

    def ddl_aware_normalize(x):
        x = np.copy(x)
        x_min = np.min(x) if N > 0 else 0.0
        x_max = np.max(x) if N > 0 else 1.0
        range_val = np.maximum(x_max - x_min, eps)
        clip_offset = 0.6594014585310611 * range_val
        x_clipped = np.clip(x, x_min - clip_offset, x_max + clip_offset)
        return (x_clipped - x_min) / (range_val + eps)
    norm_critical = ddl_aware_normalize(critical_pressure)
    norm_bottleneck = ddl_aware_normalize(bottleneck_pressure)
    norm_energy = ddl_aware_normalize(energy_term)
    wait_scaled = ready_wait_time / (1.0 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = ddl_aware_normalize(wait_saturation)
    score = neg_slack + 0.5026338047530892 * norm_critical + 0.09442926164015969 * norm_bottleneck + 1.2550763772325206 * norm_energy - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
