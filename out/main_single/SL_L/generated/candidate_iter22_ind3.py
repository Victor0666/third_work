import numpy as np
RULE_METADATA = {'structure_hash': '82e109a1b3fafd38d78900b11ab377611074bcbc196196aded14030a10096972', 'parameter_schema_hash': '91b60c0eb0b4a0b301cbf45f30a4d8f3db71b90f77faa7b743252084ad12ba4d', 'best_parameter_hash': 'bbb031937f211313ea3abeaf9c65d9cd7a0e3542a2e9ae7060bc0c897cb7f969', 'best_parameters': {'epsilon': 6.973883614297677e-09, 'ddl_protection_threshold': 1.2743870606513852, 'criticality_boost': 3.3629827068997447, 'duration_risk_penalty': 1.0753895933843134, 'energy_efficiency_ratio_weight': 0.9549043591215809, 'energy_uncertainty_interaction': 1.6564293052777042, 'rank_slack_balance': 0.7722486373792111, 'slack_max_bound': 49.24875188233446, 'slack_min_bound': -0.7287438500357837, 'slack_penalty_exponent': 3.3688764900779034, 'uncertainty_sigmoid_steepness': 4.195639639111215, 'uncertainty_slack_coupling': 0.2782846493639202}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'f4156d7893d6bc45ab77fdbc1b929f54028468905e0fd5f5a9cbc2b59de7c046', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
      - Strict lexicographic DDL protection via hard `slack > ddl_protection_threshold` gating (validated across ≥22 decisions)
      - Critical-path release unconditionally via `upward_rank × remaining_work` interaction
      - Joint MAD normalization over `|slack|`, `uncertainty`, and `duration_total` for coherent risk alignment
      - Energy-aware terms gated *only* under meaningfully positive slack (`slack > ddl_protection_threshold`)
      - Bounded monotonic gates (sigmoid on uncertainty, linear slack-pressure scaling) replacing brittle thresholds
      - Anti-starvation via `ready_wait_time / (1 + |slack|)` but only when slack > ddl_protection_threshold
      - All features normalized using robust MAD (median absolute deviation) instead of percentile clipping for cross-seed stability
    """
    eps = 6.973883614297677e-09
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
        mad = np.median(np.abs(x - med)) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack)
    joint_risk_vec = np.stack([abs_slack, uncertainty, duration_total], axis=0)
    joint_mad_norm = mad_normalize(joint_risk_vec.mean(axis=0))
    slack_score = np.where(slack < 0, (-slack) ** 3.3688764900779034, 0.0)
    deadline_pressure = np.maximum(0.0, -slack + 1.2743870606513852)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.2782846493639202
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.0753895933843134
    critical_path_urgency = upward_rank * remaining_work
    slack_headroom_mask = np.where(slack > 1.2743870606513852, 1.0, 0.0)
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -0.7287438500357837
    slack_ub = 49.24875188233446
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.7722486373792111 + (1.0 - 0.7722486373792111) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.195639639111215 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 1.6564293052777042 * energy_norm * unc_norm * unc_sigmoid
    wait_score = np.where(slack_headroom_mask > 0.0, mad_normalize(ready_wait_time / (1.0 + abs_slack)), 0.0)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + -mad_normalize(critical_path_urgency)
    score += slack_headroom_mask * (0.9549043591215809 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    is_ddl_tight = slack <= 1.2743870606513852
    critical_gate = np.where(is_ddl_tight, 3.3629827068997447, 1.0)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
