import numpy as np
RULE_METADATA = {'structure_hash': 'cce7decb2634550a896a0ad887f6117a577ed208173a61955d2b764fd2f133fe', 'parameter_schema_hash': '74921317e5b3863ac1d6a0ca0470ff16c92cd7e846f2507ad9306c4197062429', 'best_parameter_hash': 'bd65b893ca6945c2b66a41e0868d43d195a9374edb5bd20572b852acba9c8ae1', 'best_parameters': {'epsilon': 0.0024491693395026783, 'ddl_protection_gate_slope': 7.65168647311109, 'energy_sensitivity': 0.32802122251813765, 'energy_uncertainty_interaction': 0.2096608460321962, 'remaining_work_weight': 1.2226904046730978, 'slack_penalty_exponent': 1.271045130395529, 'slack_pressure_gate_steepness': 1.8078450884342319, 'uncertainty_gate_threshold': 0.18643698909726475, 'wait_decay': 0.5855129376192676, 'percentile_low': 14.758321922240672, 'percentile_high': 87.4661758632313}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '4fd27d2adbdd5063eb751a7f27634c923df2566a7f69af30d96ad17bc9c299d7', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces median-MAD with robust sign-preserving range normalization;
       introduces unified congestion signal (ready_wait_time + uncertainty) gated by DDL feasibility;
       removes redundant duration_robustness and wait_saturation_offset per counterfactual evidence;
       enforces strict sign-consistent prioritization: smaller score = higher priority;
       uses clipped linear slack pressure near violation boundary for numerical stability;
       incorporates upward_rank * remaining_work interaction only under DDL breach per diagnostic evidence."""
    eps = 0.0024491693395026783
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_range_normalize(x):
        x = np.copy(x)
        if N == 1:
            rng = eps
            mid = x[0]
        else:
            p_low = np.percentile(x, 14.758321922240672)
            p_high = np.percentile(x, 87.4661758632313)
            rng = p_high - p_low
            mid = (p_low + p_high) / 2.0
        spread = rng if rng > eps else eps
        return (x - mid) / spread
    norm_slack = robust_range_normalize(slack)
    norm_energy = robust_range_normalize(min_incremental_energy)
    norm_rank = robust_range_normalize(upward_rank)
    norm_work = robust_range_normalize(remaining_work)
    norm_wait = robust_range_normalize(ready_wait_time)
    norm_uncert = robust_range_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-7.65168647311109 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    slack_pressure = np.clip(-slack, 0.0, 1.0)
    congestion = (ready_wait_time + uncertainty) * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.5855129376192676 * ready_wait_time)
    uncert_gate = 1.0 / (1.0 + np.exp(-7.65168647311109 * (uncertainty - 0.18643698909726475)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.271045130395529 * slack_pressure) * ddl_gate
    score = +np.clip(slack_pressure, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 1.8078450884342319 * slack_pressure), -2.0, 2.0) - 0.32802122251813765 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(congestion, -2.0, 2.0) + 0.2096608460321962 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.271045130395529 * np.clip(energy_slack_penalty, -2.0, 2.0) + 1.2226904046730978 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
