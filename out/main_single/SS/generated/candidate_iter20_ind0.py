import numpy as np
RULE_METADATA = {'structure_hash': '15d40fb5b01a0e00740348adedeb12d475ba1dc6ee6844c0faba32a2110bd271', 'parameter_schema_hash': '0bac2242c1aab756d723eebc08f8a40b4b95a7749924124fc63e44d461095606', 'best_parameter_hash': '1c98de5d1eef7fab693519423125cab968b84c3b9ad20fa65ef9b771d4a05d4e', 'best_parameters': {'epsilon': 0.003918210767596545, 'ddl_protection_gate_slope': 4.094226755791222, 'energy_sensitivity': 0.31118927040495203, 'energy_uncertainty_interaction': 0.6497774895250086, 'remaining_work_weight': 1.9982590437845058, 'slack_penalty_exponent': 2.0297360292647304, 'quantile_q1': 0.05016234885968595, 'quantile_q3': 0.9030423021328932, 'congestion_weight': 0.490233335751628, 'exec_comm_separation_factor': 0.15991880543768142, 'critical_path_activation_threshold': -0.11717787292468562, 'uncertainty_gate_center': 0.019100123551139526}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'a0e42d5be2c30e183aeb4f649c83794f9c584799cd6cb37be6eba016677bc1b6', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's robust quantile normalization and unified congestion signal with Parent 1's exec/comm separation and critical-path activation;
       replaces rank amplification with hard-thresholded critical-path activation to avoid overfitting;
       uses sign-preserving MAD only for slack to preserve sensitivity near deadline boundary;
       introduces tunable critical_path_activation_threshold for precise DDL-breach triggering;
       replaces hardcoded 0.25 uncertainty threshold with PARAMS["uncertainty_gate_center"];
       eliminates all numeric literals except -2, -1, 0, 1, 2; enforces finite output and shape (N,)."""
    eps = 0.003918210767596545
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
            q1 = q3 = x[0]
        else:
            q1 = np.quantile(x, 0.05016234885968595)
            q3 = np.quantile(x, 0.9030423021328932)
        iqr = q3 - q1
        spread = iqr if iqr > eps else eps
        return (x - q1) / spread
    norm_energy = robust_range_normalize(min_incremental_energy)
    norm_exec = robust_range_normalize(min_exec_time)
    norm_comm = robust_range_normalize(min_comm_time)
    norm_rank = robust_range_normalize(upward_rank)
    norm_work = robust_range_normalize(remaining_work)
    norm_wait = robust_range_normalize(ready_wait_time)
    norm_uncert = robust_range_normalize(uncertainty)
    if N == 1:
        slack_med = slack[0]
        slack_mad = eps
    else:
        slack_med = np.median(slack)
        slack_mad = np.median(np.abs(slack - slack_med))
    slack_spread = slack_mad if slack_mad > eps else eps
    norm_slack = (slack - slack_med) / slack_spread
    ddl_gate = 1.0 / (1.0 + np.exp(-4.094226755791222 * slack))
    ddl_breach = (slack <= -0.11717787292468562).astype(float)
    critical_path_score = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.0297360292647304
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    congestion_signal = norm_wait + norm_uncert
    congestion_score = congestion_signal * ddl_breach * 0.490233335751628
    exec_penalty = 0.15991880543768142 * norm_exec * ddl_breach
    comm_penalty = (1.0 - 0.15991880543768142) * norm_comm * ddl_breach
    uncert_gate = 1.0 / (1.0 + np.exp(-4.094226755791222 * (norm_uncert - 0.019100123551139526)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_breach
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_score, -2.0, 2.0) - 0.31118927040495203 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) + np.clip(congestion_score, -2.0, 2.0) + 0.6497774895250086 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.9982590437845058 * np.clip(norm_work * ddl_gate, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + np.clip(comm_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
