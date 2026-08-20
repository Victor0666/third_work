import numpy as np
RULE_METADATA = {'structure_hash': '7aa2bc93d2ce23a8f37af072851d064b988c65a45451d53352c95d528fa7b818', 'parameter_schema_hash': '4ec9b04f66989b167fd7893302185948d01ee7fdcf0b592a0bcf34474c51ff49', 'best_parameter_hash': 'f2e7cfd42a4ec9710dbf816317f17d861d9dfa8306ab04090cd732edc4b8180d', 'best_parameters': {'epsilon': 0.00046809864373298214, 'slack_penalty_exponent': 1.026794983900167, 'criticality_scale': 0.9939006329240404, 'energy_sensitivity': 1.9997518109619832, 'duration_robustness': 0.4613722172047269, 'wait_decay': 0.6299349884344942, 'uncertainty_gate_threshold': 0.5617471980293374, 'slack_pressure_gate_steepness': 1.4252163235725195, 'remaining_work_weight': 0.29985942865266496, 'wait_saturation_offset': 3.2241945051344444e-07, 'energy_uncertainty_interaction': 0.6159066679152025, 'ddl_protection_gate_slope': 3.422038306756693}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '9ca1e53495103341c0f8c93449377ca7eeaaf007624dad2a21b6dca449109995', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's robust median-MAD normalization and slack-power penalty with Parent 1's explicit comm+uncert penalty;
       removes redundant 'comm_energy_coupling' to comply with 12-parameter limit; replaces it with strengthened duration_risk_score;
       enforces strict DDL-first semantics via dual-gated energy/uncertainty terms; clips all contributions to [-2,2] for stability."""
    eps = 0.00046809864373298214
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def median_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    norm_comm = median_mad_normalize(min_comm_time)
    ddl_gate = 1.0 / (1.0 + np.exp(-3.422038306756693 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.026794983900167
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.4252163235725195 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 0.9939006329240404 * rank_gate)
    uncert_gate = 1.0 / (1.0 + np.exp(-3.422038306756693 * (norm_uncert - 0.5617471980293374)))
    duration_risk_score = norm_duration * uncert_gate * slack_pressure * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.6299349884344942 * (ready_wait_time + 3.2241945051344444e-07))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.9997518109619832 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.4613722172047269 * np.clip(duration_risk_score, -2.0, 2.0) + 0.6159066679152025 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.29985942865266496 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
