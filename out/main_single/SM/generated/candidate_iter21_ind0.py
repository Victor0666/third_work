import numpy as np
RULE_METADATA = {'structure_hash': '23aeb538d6e52f7c23343f9c37659ecf76f477f39b5f0599dd8f23a50abadf24', 'parameter_schema_hash': 'e70a5c61ad13ed71b891210f472ca6caf98eaf9ba5944d6a2c37915c348b4aa1', 'best_parameter_hash': '7b15181af818025ddd41872a539a72688bbad1c52f804428db85e98b54486c09', 'best_parameters': {'epsilon': 0.0012734188364032727, 'slack_risk_penalty': 9.651410909179786, 'slack_urgency_gain': 3.132757022014799, 'energy_efficiency_weight': 0.3081639889926839, 'criticality_weight': 1.5787413927858636, 'bottleneck_proximity_weight': 1.7996820846881036, 'duration_uncertainty_ratio': 1.3358451851753093, 'wait_ramp_threshold': 29.318519188591655, 'ddl_feasibility_margin': 0.03626821005250818, 'robust_normalization_quantile': 0.6282792004831963, 'host_load_proxy_exponent': 0.7823092955744297, 'finite_safeguard_scale': 565977670991.0773}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'c5bf84ed48ed1747c14a55fd82f1074959bf7c0b059b6759638556ebe00d5cee', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Hard DDL feasibility mask (binary) for strict deadline-first enforcement.
      - Host-load–aware energy gating: uses sublinear (wait + uncertainty)^exponent as proxy for local host congestion.
      - Bottleneck interaction simplified to upward_rank * min(min_exec_time, min_comm_time) — avoids slack singularity and focuses on dominant data/compute bound.
      - All normalizations use robust quantile scaling; no mean/median bias.
      - Strict DDL-first ordering preserved via dominant slack_penalty and hard-gated energy scoring.
    """
    eps = 0.0012734188364032727
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
            scale = np.quantile(abs_x[finite_mask], 0.6282792004831963)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slk_norm = normalize(slk)
    rank_norm = normalize(rank + eps)
    exec_norm = normalize(exec_t + eps)
    comm_norm = normalize(comm_t + eps)
    energy_norm = normalize(energy + eps)
    wait_norm = normalize(wait + eps)
    uncert_norm = normalize(uncert + eps)
    slack_penalty = np.where(slk_norm < 0, 9.651410909179786 * np.abs(slk_norm), -3.132757022014799 * slk_norm)
    ddl_feasibility_margin = 0.03626821005250818
    ddl_feasible_mask = np.where(slk >= -ddl_feasibility_margin, 1.0, 0.0)
    host_load_proxy = np.power(wait + uncert + eps, 0.7823092955744297)
    host_load_gate = 1.0 / (1.0 + host_load_proxy)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.3081639889926839 * normalize(inv_energy) * ddl_feasible_mask * host_load_gate
    rank_score = -1.5787413927858636 * rank_norm
    min_duration_bound = np.minimum(exec_t, comm_t)
    bottleneck_interaction = rank_norm * normalize(min_duration_bound + eps)
    bottleneck_score = -1.7996820846881036 * normalize(bottleneck_interaction + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    dur_uncert_blend = dur_norm + 1.3358451851753093 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_saturation = 1.0 - np.exp(-wait / (29.318519188591655 + eps))
    wait_score = -normalize(wait_saturation + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 565977670991.0773
    min_safe = -finfo.max / 565977670991.0773
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
