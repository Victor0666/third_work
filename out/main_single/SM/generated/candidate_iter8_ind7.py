import numpy as np
RULE_METADATA = {'structure_hash': '052e785ba7002593c70f23e9d873cf74a928ffdb514cb244a0c6218e282b0484', 'parameter_schema_hash': '7c8dc1050e590b8f80aa8f7c91c694d4c667171d77c3f487b1c6a8a611d4ea4a', 'best_parameter_hash': '098b766c5b0f7a74be83e58c9139f2bf9dd1bb7c53db70e1bcc7081dbe3f2c65', 'best_parameters': {'epsilon': 0.00018763214845270057, 'slack_risk_penalty': 9.001763358029619, 'slack_urgency_gain': 2.5773533747279562, 'energy_efficiency_weight': 1.2456984181626614, 'criticality_weight': 0.5700408569669487, 'bottleneck_proximity_weight': 0.9767855778688738, 'duration_uncertainty_ratio': 0.00151268339315788, 'wait_clip_threshold': 23.160003461554624, 'ddl_protection_gate': 0.021506987826148162, 'finfo_max_scale': 11409017.997114716, 'robust_normalization_quantile': 0.7321893757578894}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '65d395065686895a3d7154e6ee25f3ee29f5cc091c0157bc32cec38241943ab9', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Retains smooth sigmoid ddl_protection_gate (Parent 2) for graded feasibility enforcement
      - Keeps explicit criticality_weight AND bottleneck_proximity_weight (Parent 2) for decoupled control
      - Upgrades normalization from median to adaptive quantile-based scaling (novel improvement)
      - Uses robust quantile (e.g., 75th percentile of abs values) instead of median for better sensitivity
        to right-skewed outlier distributions common in cloud-edge execution times
      - Preserves hard wait-time clipping and bottleneck-proximal term (upward_rank * remaining_work)
      - All features normalized compatibly; no exponential saturations or fragile interactions
      - Strictly enforces DDL-first feasibility via slack_penalty dominating final score
    """
    eps = 0.00018763214845270057
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], 0.7321893757578894)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.021506987826148162 * slack_norm))
    slack_penalty = np.where(slk < 0, 9.001763358029619 * np.abs(slack_norm), -2.5773533747279562 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.2456984181626614 * normalize(inv_energy)
    rank_score = -0.5700408569669487 * ddl_gate * normalize(rank + eps)
    bottleneck = rank * work
    bottleneck_score = -0.9767855778688738 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.00151268339315788 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 23.160003461554624)
    wait_score = -normalize(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 11409017.997114716
    min_safe = -finfo.max / 11409017.997114716
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
