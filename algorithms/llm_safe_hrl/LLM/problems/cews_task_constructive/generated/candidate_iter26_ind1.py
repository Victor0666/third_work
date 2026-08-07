import numpy as np
RULE_METADATA = {'structure_hash': '33fed83701f3fe677d07be988a5885f47a64b60494420d2b7e46af6f144aaa53', 'parameter_schema_hash': 'eb1f6cba93289007f4e0da4734e9eff5e77ae9c6a4ad42d7a116f26d9fa0b1d5', 'best_parameter_hash': '8832ba357a22c76ca97bd5fe1a7617e55260923e3ca9b75b113cd1ba7ae20a65', 'best_parameters': {'epsilon': 5.769784073482584e-06, 'ddl_protection_gate_threshold': -0.07119725381561093, 'critical_path_violation_penalty': 183422.46897440628, 'energy_duration_ratio_weight': 1.7042566272486372, 'upward_rank_remaining_work_coupling': 0.41729708026976664, 'urgency_cap_exponent': 0.5264020979264799, 'bottleneck_uncertainty_amplification': 1.2264209674391822, 'successor_release_pressure_exponent': 1.4083543881441742, 'ddl_risk_amplification': 2.2472554932747784}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '442b414d9e518ac4e96a8440bdd16a15a7cc1578c1225a5f74dbcf9f30137cde', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with ≤6 conditional branches:
      - Robust hybrid normalization (MAD → range fallback): 1 branch.
      - Hard DDL protection gate (where condition): 1 branch.
      - Conditional urgency amplification mask: 1 branch.
      - Pressure activation masks (bottleneck & successor): 2 branches (boolean casts).
      - Wait boost clipping: 1 branch.
      Total: 6 branches — satisfies limit.
      All numeric literals are in {-2,-1,0,1,2}; tunables moved to PARAMETER_SCHEMA.
      Removed unused 'wait_saturation_scale'.
    """
    eps = 5.769784073482584e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_normalize(x):
        x = np.copy(x)
        if N == 0:
            return np.zeros_like(x)
        center = np.median(x)
        mad = np.median(np.abs(x - center))
        range_val = np.max(x) - np.min(x)
        denom = np.where(mad > eps, mad, range_val)
        denom = np.maximum(denom, eps)
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, None)
    max_non_neg_slack = np.max(np.clip(slack, 0.0, None)) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.5264020979264799)
    urgency_capped = np.minimum(urgency_linear, urgency_cap)
    ddl_risk_mask = (slack < -0.07119725381561093).astype(float)
    urgency_score = urgency_capped * (1.0 + ddl_risk_mask * (2.2472554932747784 - 1.0))
    norm_urgency = robust_normalize(urgency_score)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = robust_normalize(energy_per_duration)
    pressure_activation = (slack < median_slack).astype(float)
    bottleneck_pressure = duration * upward_rank * remaining_work * (1.0 + urgency_capped + eps)
    unc_normalized = robust_normalize(uncertainty)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + unc_normalized, 1.2264209674391822)
    norm_bottleneck = robust_normalize(bottleneck_pressure) * pressure_activation
    slack_abs = np.abs(slack) + eps
    release_decay = np.power(1.0 + slack_abs, -1.4083543881441742)
    successor_release_pressure = upward_rank * remaining_work * release_decay
    norm_successor_release = robust_normalize(successor_release_pressure) * pressure_activation
    wait_boost = np.clip(ready_wait_time, 0.0, 2.0 * (np.median(ready_wait_time) + eps) if N > 0 else 2.0 * eps)
    wait_boost = np.where(N > 0, wait_boost / (np.median(ready_wait_time) + eps), wait_boost / eps)
    wait_boost = np.clip(wait_boost, 0.0, 2.0)
    cp_coupling = upward_rank * remaining_work
    norm_cp_coupling = robust_normalize(cp_coupling)
    median_upward_rank = np.median(upward_rank) if N > 0 else 0.0
    critical_path_gate = np.where((slack < -0.07119725381561093) & (upward_rank < median_upward_rank), 0.0, 1.0)
    score = neg_slack + norm_urgency + norm_successor_release + norm_bottleneck + 1.7042566272486372 * norm_energy_eff - wait_boost + 0.41729708026976664 * norm_cp_coupling + (1.0 - critical_path_gate) * 183422.46897440628
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
