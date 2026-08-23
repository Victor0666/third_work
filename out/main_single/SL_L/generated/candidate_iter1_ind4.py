import numpy as np
RULE_METADATA = {'structure_hash': 'a57da5709483629d3ada1b70cb12d35c2e30c275fb5b932fb2035c1c2c4a22c1', 'parameter_schema_hash': 'e8d5b5115ab051e9ec4d07c73e4c72851380c65f5b66e4065c779972f41c5095', 'best_parameter_hash': '738417cd6aa2916c14c10618a766cc46ea41f2ebcc3d5c75cc47b3b108f7085f', 'best_parameters': {'epsilon': 4.376503733603261e-05, 'slack_risk_exponent': 2.126114964479193, 'criticality_gate_threshold': 0.16209210269885088, 'energy_sensitivity': 2.621751546079255, 'duration_uncertainty_coupling': 1.536808471707451, 'wait_decay_rate': 0.07310143862240713, 'rank_fallback_weight': 1.1886428613474949, 'uncertainty_slack_interaction': 0.1955267557628772, 'wait_boost_weight': 0.15375983353997527, 'nan_replacement': -269889.13545355236, 'posinf_clip': 27843156.88361157, 'neginf_clip': -492726663.6557073}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'e46d6539faa1e8160fa5ac6507ae29796b2692e855431f8891446587610132f1', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: neginf_clip now uses identity transform.
    Uses risk-gated criticality, bounded uncertainty-duration coupling, and exponential wait boost.
    Smaller score = higher priority. All numeric literals are -2,-1,0,1,2 or derived from PARAMS."""
    eps = 4.376503733603261e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_norm(x):
        x_abs = np.abs(x)
        denom = np.mean(x_abs) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    norm_duration = robust_norm(min_exec_time + min_comm_time)
    norm_rank = robust_norm(upward_rank)
    norm_work = robust_norm(remaining_work)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    slack_power = np.sign(norm_slack) * np.abs(norm_slack) ** 2.126114964479193
    slack_normalized_for_gate = (norm_slack - 0.16209210269885088) / (1.0 + eps)
    gate_activation = np.clip(-slack_normalized_for_gate, 0.0, 1.0)
    dur_uncert_coupling = np.clip(norm_duration * norm_uncert, 0.0, 1.0) * 1.536808471707451
    wait_boost = 1.0 - np.exp(-0.07310143862240713 * (norm_wait + eps))
    slack_pressure_mask = np.where(norm_slack < 0, 1.0, 0.0)
    uncert_slack_penalty = norm_uncert * slack_pressure_mask * 0.1955267557628772
    score = slack_power
    safe_slack_mask = np.where(norm_slack > 0, 1.0, 0.0)
    score += safe_slack_mask * 2.621751546079255 * norm_energy
    score -= gate_activation * 1.1886428613474949 * norm_rank
    score += dur_uncert_coupling
    score -= wait_boost * 0.15375983353997527
    score += uncert_slack_penalty
    score = np.nan_to_num(score, nan=-269889.13545355236, posinf=27843156.88361157, neginf=-492726663.6557073)
    return score
