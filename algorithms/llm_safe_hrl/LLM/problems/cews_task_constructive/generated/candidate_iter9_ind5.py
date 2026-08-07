import numpy as np
RULE_METADATA = {'structure_hash': '69621a2338d9c8fcae031a27547982631ef4e9ad0b1ac349ac28cdf607b7a97c', 'parameter_schema_hash': 'f948979823dd951772b6a1a0f3ae226d34a308591cf64ba6d7259282ce584896', 'best_parameter_hash': '655772ce1ae35d448a3e16b8c6cd68bc2d37714ea45e3d1bc7b6e53eddb72b59', 'best_parameters': {'epsilon': 0.0533310788073146, 'energy_duration_ratio_weight': 0.7314727610863714, 'successor_bottleneck_coupling': 0.574542560106594, 'iqr_low_percentile': 32.66877893095063, 'iqr_high_percentile': 66.15288562367957, 'ddl_risk_threshold': -0.6980934225157416, 'bottleneck_work_exponent': 0.5346501411979188, 'wait_clip_threshold': 115.57378738921938}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '6e2a6f764b6fc4c6015a8e0fdcad171bef5bcce8e540d4089ad6361b91d1b8ee', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces fragile sigmoid urgency with monotonic bounded linear ramp,
    reintroduces remaining_work into bottleneck coupling via power-law scaling to prevent starvation,
    eliminates inactive parameters (slack_sigmoid_*, bottleneck_slack_modulation, sigmoid_clip_bound),
    and replaces percentile-based DDL-protection gate with hard slack threshold + uncertainty gating.
    
    Key structural improvements:
      - Bounded linear urgency: max(0, -slack - PARAMS["ddl_risk_threshold"]) ensures monotonic,
        interpretable, and numerically stable deadline pressure — no exp overflow or saturation artifacts.
      - Bottleneck coupling now includes remaining_work^exponent to preserve work-volume signal while avoiding dominance;
        avoids redundancy with upward_rank yet captures descendant load pressure.
      - DDL-protection gate uses absolute slack threshold (not median-relative) + uncertainty > median_uncertainty,
        improving robustness under sparse or skewed slack distributions.
      - All normalization remains IQR-based with elite-tuned percentiles; no percentile gating or arctan.
      - Critical bonus and anti-starvation terms preserved but simplified: direct rank boost and clipped log wait.
    """
    eps = 0.0533310788073146
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
        q_low = np.percentile(x, 32.66877893095063)
        q_high = np.percentile(x, 66.15288562367957)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    urgency_base = -slack - -0.6980934225157416
    urgency_penalty = np.maximum(urgency_base, 0.0) + eps
    norm_urgency = iqr_normalize(urgency_penalty)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    work_scaled = np.power(np.maximum(remaining_work, eps), 0.5346501411979188)
    bottleneck_pressure = duration * (upward_rank + eps) * work_scaled
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = norm_rank
    wait_clipped = np.clip(ready_wait_time, 0.0, 115.57378738921938)
    wait_log = np.log1p(wait_clipped)
    norm_wait = iqr_normalize(wait_log)
    median_uncertainty = np.median(uncertainty) if N > 0 else 0.0
    ddl_protection_active = ((slack <= -0.6980934225157416) & (uncertainty > median_uncertainty)).astype(float)
    score = norm_urgency + 0.574542560106594 * norm_bottleneck + 0.7314727610863714 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_active * norm_urgency
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
