import numpy as np
RULE_METADATA = {'structure_hash': 'fe7c70bf923e63f604fdd270761fa28fe5db1baf6ffa56f9185e5b26ed18f138', 'parameter_schema_hash': '3d8273baab52881e917be1b16b18703175798bb1dc6d2292105548cef6083b8d', 'best_parameter_hash': '425f55060704536f016e39ba6509097593af3d3f306068986c9598c646a9c934', 'best_parameters': {'epsilon': 4.804038037220669e-06, 'slack_penalty_exponent': 1.1176739099626174, 'criticality_boost': 1.4630175744337839, 'energy_efficiency_ratio_weight': 1.6611928570920507, 'uncertainty_slack_coupling': 2.5863588073817025, 'rank_slack_balance': 0.37396026411668337, 'duration_risk_penalty': 1.0055404490152702, 'energy_uncertainty_interaction': 0.2439905439659376, 'uncertainty_sigmoid_steepness': 0.7530119484708754, 'ddl_protection_threshold': 3.8864652065762444, 'upward_rank_remaining_work_interaction': 1.6382298046689465, 'host_load_conditional_gate': 1.657280585197676}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'b1fbd9adfa7ec85fc91394b0390abdf80ea035c6c8708b31f3cb81ef13cfe8b2', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating key counterfactual evidence:
      - Replaces strict 'slack > 0' gate with tighter 'slack > ddl_protection_threshold' to prevent premature energy optimization
      - Adds explicit upward_rank × remaining_work interaction as successor-release signal (confirmed in 6+ starvation failures)
      - Introduces host-load conditional gate: energy terms only activated when energy is below a normalized threshold
      - Uses MAD-based global feature scaling instead of per-feature percentile normalization for rank stability
      - Replaces ready_wait_time term with monotonic, bounded inverse slack weighting to avoid numerical explosion
      - All DDL-critical terms remain unconditionally active; non-DDL terms now require both slack headroom AND low energy load
      - Criticality boost applied multiplicatively only when slack <= 0 AND upward_rank × remaining_work is above median
      - No fragile sigmoid/percentile thresholds: all gates use bounded, monotonic forms
      - Final score preserves lexicographic DDL-first ordering while enabling safer energy optimization
    """
    eps = 4.804038037220669e-06
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

    def global_mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 1.1176739099626174, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.5863588073817025
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.0055404490152702
    rank_work_product = upward_rank * remaining_work
    rank_work_med = np.median(rank_work_product) if N > 1 else np.mean(rank_work_product)
    is_high_critical_path = rank_work_product >= rank_work_med
    criticality_boost_mask = np.where((slack <= 0) & is_high_critical_path, 1.4630175744337839, 1.0)
    slack_headroom_mask = np.where(slack > 3.8864652065762444, 1.0, 0.0)
    energy_norm = global_mad_normalize(min_incremental_energy)
    load_gate = np.where(energy_norm < 1.657280585197676, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = global_mad_normalize(energy_per_sec)
    inv_slack = np.where(slack > 3.8864652065762444, 1.0 / (slack + 1.0), 0.0)
    wait_score = global_mad_normalize(ready_wait_time * inv_slack)
    rank_score = -global_mad_normalize(upward_rank)
    slack_scaled = np.clip((slack - 3.8864652065762444) / (3.8864652065762444 + 1.0), 0.0, 1.0)
    weight_rank = 0.37396026411668337 + (1.0 - 0.37396026411668337) * (1.0 - slack_scaled)
    rank_score = rank_score * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-0.7530119484708754 * (uncertainty - 1.0)))
    energy_uncertainty_score = 0.2439905439659376 * energy_norm * global_mad_normalize(uncertainty) * unc_sigmoid
    score = global_mad_normalize(slack_score) + global_mad_normalize(unc_slack_coupling) + global_mad_normalize(duration_risk)
    score += slack_headroom_mask * load_gate * (1.6611928570920507 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score + 1.6382298046689465 * global_mad_normalize(rank_work_product))
    score = score * criticality_boost_mask
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
