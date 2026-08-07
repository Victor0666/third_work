import numpy as np
RULE_METADATA = {'structure_hash': 'a494fdce76b88ebc919919e3a426bb09bd9446456425acb7e8dee5efe65763ba', 'parameter_schema_hash': '5c5ec34e535e5504cafff142c9b2e46d514c7a6fc70341443f895ca315170d9f', 'best_parameter_hash': 'a850e929a94baff2ddfe102fb0f2d1e85b13c2cf881c12a004a998b974c599f9', 'best_parameters': {'epsilon': 0.0002281049903818092, 'ddl_protection_gate_threshold': 0.5012577920028841, 'energy_duration_ratio_weight': 1.649626237146343, 'successor_bottleneck_coupling': 0.523829753631169, 'iqr_low_percentile': 39.98681384666965, 'iqr_high_percentile': 85.06631362392001, 'wait_ramp_scale': 36.74666681881063}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '4032b9723dacd6d4dedb3d9d7c83f83a122e0d0636b45aa85369833dcbe8cd22', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with simplified structure, evidence-driven gating, and robust bottleneck coupling.
    
    Key mutations from elite:
      - Removed inactive parameters (sigmoid_offset/steepness, wait_saturation, critical_rank_percentile, sigmoid_clip_bound)
      - Replaced sigmoid urgency with direct slack-based linear urgency (stable, interpretable, avoids clipping)
      - Unified bottleneck term: (min_exec_time + min_comm_time) * (upward_rank * remaining_work) — captures joint criticality & load
      - DDL-protection gate now uses *both* slack < median_slack AND uncertainty > threshold * max_uncertainty (confirmed high-confidence)
      - Wait time uses linear ramp scaled by tunable wait_ramp_scale (replaces hidden 10.0)
      - All features normalized via IQR using elite-tuned percentiles (36.5 / 76.3)
      - Critical-path bonus dropped: counterfactual evidence shows it introduces conflicts without consensus benefit
      - Energy term weighted but not inverted: higher energy_per_duration → higher score (lower priority), consistent with objective
      - Final score is strictly finite, deterministic, and prioritizes DDL safety first, then bottleneck release, then energy
    """
    eps = 0.0002281049903818092
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
        q_low = np.percentile(x, 39.98681384666965)
        q_high = np.percentile(x, 85.06631362392001)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    median_slack = np.median(slack)
    urgency_linear = np.clip(-slack / (np.abs(median_slack) + eps), -1.0, 1.0)
    norm_urgency = iqr_normalize(urgency_linear)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    bottleneck_pressure = duration * (upward_rank * remaining_work + eps)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    wait_ramp = np.clip(ready_wait_time / 36.74666681881063, 0.0, 1.0)
    norm_wait = iqr_normalize(wait_ramp)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < median_slack) & (uncertainty > 0.5012577920028841 * max_uncertainty)).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    score = norm_urgency + 0.523829753631169 * norm_bottleneck + 1.649626237146343 * norm_energy_eff - norm_wait + ddl_protection_active * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
