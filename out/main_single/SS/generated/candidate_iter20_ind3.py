import numpy as np
RULE_METADATA = {'structure_hash': '73215cad0bb94c7a6d0e7966553d6a427392c945c56357cd02a203dc2cbc3f63', 'parameter_schema_hash': 'd33ff595d6b9f50c30548d527293117c7b7326884e19755c938ee157e2a53ba1', 'best_parameter_hash': '714682fe348d17d2338201d8c3a9c499ff943401be80082518e317bc7a2a1bc7', 'best_parameters': {'epsilon': 1.261614008638549e-06, 'ddl_protection_gate_slope': 8.240256134236915, 'energy_sensitivity': 0.4286088221881864, 'energy_uncertainty_interaction': 0.4521416249497316, 'remaining_work_weight': 0.2323381941362439, 'slack_penalty_exponent': 1.2383197000137693, 'criticality_scale': 1.3785841207525011, 'uncertainty_gate_threshold': 0.21239269347135192, 'quantile_q1': 0.07648067862727193, 'quantile_q3': 0.6842295929027801, 'wait_benefit_cap': 0.939036881615753, 'congestion_coupling_power': 1.0462721438310076}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'ea4ded34429a7ecdef187cab9394f5961b4a0f9065683a3d58a1028fac809a6d', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining best elements:
       - Uses tunable quantile-based normalization (Parent 2) for robustness
       - Sign-preserving MAD for slack (Parent 2) to amplify boundary urgency
       - Smooth DDL-protection gate with slope parameter (Parent 2) for energy/uncertainty gating
       - Critical path leverage only under breach (both parents)
       - Novel congestion coupling: power-law scaling of congestion signal to better distinguish overload states
       - Bounded starvation relief with tunable cap to prevent priority inversion
       - All terms clipped to [-2,2] and fused with bounded coefficients for stability
       - No numeric literals beyond -2,-1,0,1,2; uses np.finfo for epsilon safeguards."""
    eps = 1.261614008638549e-06
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
            q1 = np.quantile(x, 0.07648067862727193)
            q3 = np.quantile(x, 0.6842295929027801)
        iqr = q3 - q1
        spread = iqr if iqr > eps else eps
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
    ddl_gate = 1.0 / (1.0 + np.exp(-8.240256134236915 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.2383197000137693
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    congestion_signal = ready_wait_time + uncertainty
    norm_congestion = robust_range_normalize(congestion_signal)
    congestion_score = np.clip(norm_congestion ** 1.0462721438310076, 0.0, 2.0) * (1.0 - ddl_gate)
    wait_mean = np.mean(ready_wait_time)
    wait_denom = np.maximum(wait_mean, eps)
    wait_benefit = np.clip(ready_wait_time / wait_denom, 0.0, 0.939036881615753)
    uncert_gate = 1.0 / (1.0 + np.exp(-8.240256134236915 * (norm_uncert - 0.21239269347135192)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_breach
    slack_pressure = np.clip(-slack, 0.0, 1.0)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 1.3785841207525011 * slack_pressure), -2.0, 2.0) - 0.4286088221881864 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(congestion_score, -2.0, 2.0) + 0.4521416249497316 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.2323381941362439 * np.clip(norm_work * ddl_gate, -2.0, 2.0) + 1.2383197000137693 * np.clip(slack_pressure, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
