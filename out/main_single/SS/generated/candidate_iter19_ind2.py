import numpy as np
RULE_METADATA = {'structure_hash': 'fb5ea22cdbc3c346e7d479ace24797445bdf10c86840329ee1705cdcc3e7ede5', 'parameter_schema_hash': '309fc16e02357a8e36b9260aa3b19166bd015dca7e3c6e74dc24caf7a43e98e8', 'best_parameter_hash': '551e1a882a3431572419e13267079a1ac965d047ff14e23327710585f27525da', 'best_parameters': {'epsilon': 7.859045282159187e-05, 'ddl_protection_gate_slope': 4.127830132523045, 'energy_sensitivity': 0.7290946262503604, 'energy_uncertainty_interaction': 0.8017389359051335, 'remaining_work_weight': 0.779877360015866, 'slack_penalty_exponent': 2.53019797217243, 'criticality_scale': 1.1106367215100095, 'wait_decay': 0.47755754584929383, 'uncertainty_gate_threshold': 0.08664744876698471, 'rank_boost_lower_bound': 0.15953911696596018, 'rank_boost_upper_bound': 2.4063314579875157}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'bb173fc35eb098ff3866a87c1de4f6b5b2189be5f96a37d1fec1fa9de3983df9', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: replaces hidden 0.5 with PARAMS["rank_boost_lower_bound"];
       uses robust range-based normalization;
       introduces unified congestion signal (ready_wait_time + uncertainty) gated by DDL feasibility;
       enforces critical path leverage only under strict slack <= 0;
       replaces multiplicative ddl_gate with additive binary gate for numerical stability;
       all terms clipped to [-2,2] and fused with bounded coefficients."""
    eps = 7.859045282159187e-05
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
            xmin = xmax = x[0]
        else:
            xmin = np.min(x)
            xmax = np.max(x)
        spread = xmax - xmin if xmax - xmin > eps else eps
        return (x - xmin) / spread
    norm_slack = robust_range_normalize(slack)
    norm_energy = robust_range_normalize(min_incremental_energy)
    norm_rank = robust_range_normalize(upward_rank)
    norm_work = robust_range_normalize(remaining_work)
    norm_wait = robust_range_normalize(ready_wait_time)
    norm_uncert = robust_range_normalize(uncertainty)
    congestion = ready_wait_time + uncertainty
    norm_congestion = robust_range_normalize(congestion)
    ddl_gate = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_gate
    slack_pressure = np.clip(1.0 - norm_slack, 0.0, 1.0)
    wait_benefit = 1.0 - np.exp(-0.47755754584929383 * ready_wait_time)
    congestion_gate = 1.0 / (1.0 + np.exp(-4.127830132523045 * (norm_congestion - 0.08664744876698471)))
    energy_uncert_penalty = norm_energy * norm_uncert * congestion_gate * ddl_gate
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.53019797217243
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    rank_boost = 1.0 + 1.1106367215100095 * slack_pressure
    rank_boost = np.clip(rank_boost, 0.15953911696596018, 2.4063314579875157)
    boosted_rank = norm_rank * rank_boost
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) - 0.7290946262503604 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) + 0.8017389359051335 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.779877360015866 * np.clip(norm_work, -2.0, 2.0) + np.clip(norm_congestion * ddl_gate, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
