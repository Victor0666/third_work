import numpy as np
RULE_METADATA = {'structure_hash': '0ebda00b4025a678cb554dad9bf0a1eddee6680d147702fe37bea48742390265', 'parameter_schema_hash': 'bb15019325543fdc76211c26c988e922343270008cda783e2c2a1faa7435ba72', 'best_parameter_hash': '2b5f9240daddd844109842bdb3a52c7f37a77504f693b661d32394d529700b16', 'best_parameters': {'epsilon': 3.8062822197043984e-06, 'slack_penalty_exponent': 3.9969323965343984, 'criticality_boost': 1.9197714750046213, 'energy_sensitivity': 0.573705308523047, 'duration_balance': 0.16577298753904976, 'wait_decay_rate': 0.08591525349029643, 'uncertainty_slack_coupling': 0.5607545171921138, 'rank_energy_interaction': 0.8998927371333341, 'urgency_clip_max': 5.478126233979717}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'c10ed7f609b8230ac58ac6caa967c01fbe6f5cdb560fb917fbe961cf80d580e5', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Novel priority rule: prioritizes deadline-criticality first via robust slack gating,
    couples uncertainty with slack to amplify risk awareness, uses exponential urgency for negative slack,
    applies conditional criticality boost, and penalizes high-energy + high-rank combinations.
    Anti-starvation uses soft exponential wait boost instead of linear scaling.
    All features are robustly normalized using mean-abs + epsilon; no unbounded ops.
    """
    eps = 3.8062822197043984e-06

    def robust_norm(x):
        x = np.asarray(x, dtype=np.float64)
        denom = np.mean(np.abs(x)) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    norm_duration = robust_norm(min_exec_time + min_comm_time)
    norm_rank = robust_norm(upward_rank)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    slack_sign_mask = (slack < 0).astype(float)
    urgency_base = -norm_slack * slack_sign_mask
    urgency_clipped = np.clip(urgency_base, 0.0, 5.478126233979717)
    urgency_penalty = np.exp(3.9969323965343984 * urgency_clipped)
    rank_median = np.median(norm_rank)
    critical_mask = (norm_rank >= rank_median).astype(float)
    critical_boost = 1.9197714750046213 * norm_rank * critical_mask
    slack_pressure = np.maximum(0.0, -norm_slack)
    uncertainty_coupled_pressure = 0.5607545171921138 * slack_pressure * norm_uncert
    wait_boost = 1.0 - np.exp(-0.08591525349029643 * norm_wait)
    rank_energy_penalty = 0.8998927371333341 * norm_energy * norm_rank
    score = urgency_penalty + uncertainty_coupled_pressure + rank_energy_penalty + 0.573705308523047 * norm_energy + 0.16577298753904976 * norm_duration - critical_boost - wait_boost
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=-finfo.max)
    return score
