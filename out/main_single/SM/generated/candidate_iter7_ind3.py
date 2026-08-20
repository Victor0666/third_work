import numpy as np
RULE_METADATA = {'structure_hash': '707fa40f1d1cba2f6eb2b243aaaaf8075715affeef7aa1edffdf0c879e796722', 'parameter_schema_hash': 'd831f0524a9dc4d2374728a20a494b9e82bca2908c233276d64416bbe18bd7d7', 'best_parameter_hash': 'e7301b4385af2c96a22bf4e5ff7286c2cbb99b3041accff5bfc1b98a0abfd86a', 'best_parameters': {'epsilon': 0.004171680441375862, 'slack_risk_penalty': 7.043892898361592, 'slack_urgency_gain': 1.1739544027720128, 'energy_efficiency_weight': 1.8095574393674616, 'criticality_weight': 1.7927444610737844, 'duration_uncertainty_ratio': 0.11928484367859285, 'wait_clip_threshold': 14.777867551991202, 'bottleneck_proximity_weight': 0.8769229794147634, 'ddl_protection_gate': 0.03214808399841542, 'finfo_max_scale': 93589977.3780295}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'c1cad7b6f0e27b8f406037de598576c591f8bb5955a2a53d57188a36c8819880', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with:
      - Hard wait-time clipping instead of exponential decay (more stable & interpretable)
      - Bottleneck-proximal term: upward_rank * remaining_work → identifies high-impact critical-path bottlenecks
      - Smooth sigmoid DDL protection gate → softly suppresses non-urgent tasks without hard thresholds
      - Removed inactive uncertainty-slack interaction per evidence; replaced with robust bottleneck term
      - All normalizations use median-based scaling (more outlier-resilient than mean-abs)
      - Strict adherence to feasibility-first: no energy terms dominate negative-slack tasks
    """
    eps = 0.004171680441375862
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
        scale = np.median(abs_x[np.isfinite(abs_x)]) if np.any(np.isfinite(abs_x)) else eps
        scale = np.where(scale > eps, scale, eps)
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.03214808399841542 * slack_norm))
    slack_penalty = np.where(slk < 0, 7.043892898361592 * np.abs(slack_norm), -1.1739544027720128 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.8095574393674616 * normalize(inv_energy)
    rank_score = -1.7927444610737844 * ddl_gate * normalize(rank + eps)
    bottleneck = rank * work
    bottleneck_score = -0.8769229794147634 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.11928484367859285 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 14.777867551991202)
    wait_score = -normalize(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 93589977.3780295
    min_safe = -finfo.max / 93589977.3780295
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
