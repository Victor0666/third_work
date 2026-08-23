import numpy as np
RULE_METADATA = {'structure_hash': 'd7af05b6d6a48644d512236bd5b8e8f1434b05edda067edd3d44f8454d30ebac', 'parameter_schema_hash': '34c9d467fb160262adbcd5b227419b62c7dbea2a17330b2b3b0d70d20707d52f', 'best_parameter_hash': 'fbfb7e296bd0a7504969833c0e73d2efbe2b905ba25562125ffa781d622f8854', 'best_parameters': {'epsilon': 6.461325163912754e-08, 'slack_penalty_exponent': 3.487864221551905, 'criticality_boost': 1.3094180189603728, 'energy_efficiency_ratio_weight': 0.37062172452977615, 'uncertainty_slack_coupling': 0.6593235469998242, 'rank_slack_balance': 0.5882554193466525, 'duration_risk_penalty': 0.3980233254774681, 'energy_uncertainty_interaction': 1.0736038588752321, 'uncertainty_sigmoid_steepness': 5.864427063563453, 'slack_min_bound': -13.1010181552559, 'slack_max_bound': 46.7334380030792, 'wait_time_decay_exponent': 1.0644580978362055}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '88c3a2d626d16d840ced4c9c90ea72005851d1837c75bd66161f7d0af7cacaf4', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
      - Strict lexicographic DDL gating via hard `slack > 0` mask (simplified to binary threshold)
      - Critical path release prioritization: `upward_rank * remaining_work` interaction scaled by criticality_boost under DDL pressure
      - Power-law anti-starvation: `ready_wait_time ** wait_time_decay_exponent`, gated by slack headroom and MAD-normalized
      - MAD-based normalization per dimension instead of percentile (more stable for small N)
      - Explicit slack sign-aware scaling: linear interpolation from 0 to 1 over [slack_min_bound, slack_max_bound] for rank balance
      - All divisions guarded by eps; all inf/nan replaced before computation; no in-place mutation
      - Final score structure: [DDL violation penalty] + [critical path urgency] + [gated non-DDL efficiency & fairness]
    """
    eps = 6.461325163912754e-08
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        dev = np.abs(x - med)
        mad = np.median(dev) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 3.487864221551905, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.6593235469998242
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.3980233254774681
    is_ddl_constrained = slack <= 0.0
    successor_release_score = upward_rank * remaining_work * 1.3094180189603728
    successor_release_norm = -mad_normalize(successor_release_score)
    successor_release_contribution = np.where(is_ddl_constrained, successor_release_norm, 0.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -13.1010181552559
    slack_ub = 46.7334380030792
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.5882554193466525 + (1.0 - 0.5882554193466525) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-5.864427063563453 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 1.0736038588752321 * energy_norm * unc_norm * unc_sigmoid
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 1.0644580978362055, 0.0)
    wait_score = mad_normalize(wait_power)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + successor_release_contribution
    score += slack_headroom_mask * (0.37062172452977615 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
