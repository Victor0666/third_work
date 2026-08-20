import numpy as np
RULE_METADATA = {'structure_hash': '8970fa1a188935895874a5068319a9d7d433ddc8fb518c78dd742522f79f2832', 'parameter_schema_hash': '7810a3dbcb0eb5f5e734474e66d8041127549ec4f3747b5bfb3fed0e425c6e77', 'best_parameter_hash': 'b6f4aab137c2d0b7ee2a214824cd283da3b81529d98574442e3a53c3821394ae', 'best_parameters': {'epsilon': 1.0197215500732706e-05, 'ddl_protection_gate_slope': 2.67496797840416, 'energy_sensitivity': 0.10178554503188024, 'energy_uncertainty_interaction': 0.015795309788090954, 'remaining_work_weight': 1.2400230312389857, 'slack_penalty_exponent': 1.3698831226597221, 'criticality_scale': 2.243855777978241, 'uncertainty_gate_threshold': 0.8165655952970938, 'quantile_q1': 0.19876866028227913, 'quantile_q3': 0.7241059425180632, 'wait_benefit_cap': 0.7124448086683461, 'congestion_weight': 0.7523754668960623}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '4e2932eb7fc949100a533a56da81e089a5f7a85f1545ca6f220e1902b0ac5d9d', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: combines Parent 2's robust quantile normalization and uncertainty gating with Parent 1's clean congestion logic;
       introduces wait_benefit_cap to prevent starvation overcompensation;
       replaces raw slack pressure with sign-preserving MAD-normalized slack for deadline-boundary sensitivity;
       uses *multiplicative* criticality amplification: (1 + criticality_scale * |norm_slack|) * norm_rank * ddl_breach for sharper urgency coupling;
       enforces strict [-2,2] clipping per term to guarantee bounded AST depth and stability;
       all numeric literals are -2,-1,0,1,2 or np.finfo safeguards; no unbounded operations."""
    eps = 1.0197215500732706e-05
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
            q1 = np.quantile(x, 0.19876866028227913)
            q3 = np.quantile(x, 0.7241059425180632)
        spread = q3 - q1 + eps
        return (x - q1) / spread
    norm_energy = robust_range_normalize(min_incremental_energy)
    norm_duration = robust_range_normalize(min_exec_time + min_comm_time)
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
    ddl_gate = 1.0 / (1.0 + np.exp(-2.67496797840416 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * (1.0 + 2.243855777978241 * np.abs(norm_slack)) * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.3698831226597221
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    congestion_signal = norm_wait + norm_uncert
    congestion_score = congestion_signal * (1.0 - ddl_gate)
    wait_mean = np.mean(ready_wait_time)
    wait_denom = np.maximum(wait_mean, eps)
    wait_benefit = np.clip(ready_wait_time / wait_denom, 0.0, 0.7124448086683461)
    uncert_gate = 1.0 / (1.0 + np.exp(-2.67496797840416 * (norm_uncert - 0.8165655952970938)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_breach
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - 0.10178554503188024 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.7523754668960623 * np.clip(congestion_score, -2.0, 2.0) + 0.015795309788090954 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.2400230312389857 * np.clip(norm_work * ddl_gate, -2.0, 2.0) + np.clip(norm_rank * (1.0 - ddl_gate), -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
