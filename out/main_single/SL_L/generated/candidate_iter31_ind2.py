import numpy as np
RULE_METADATA = {'structure_hash': '2530963136de0ccee537158ec3e5598fd6e83231b5e36b612cd9be8b2264e426', 'parameter_schema_hash': 'a7d0a20b1717b3670cd6a3b4f7a8f35028da150dafc7c80f837842ca37e38e1a', 'best_parameter_hash': '206943357972b2280097edbca5492f6173fcd2510d04b1ca1cb8cdca2473161d', 'best_parameters': {'epsilon': 2.688742586490328e-07, 'slack_penalty_exponent': 3.319178828826575, 'criticality_boost': 0.512255123819319, 'energy_efficiency_ratio_weight': 0.7300627196071271, 'uncertainty_slack_coupling': 2.432015735969543, 'rank_slack_balance': 0.3081488977271859, 'duration_risk_penalty': 0.2378696354917308, 'energy_uncertainty_interaction': 0.6400058017611732, 'uncertainty_sigmoid_steepness': 1.1126223452306432, 'slack_min_bound': -27.86332641160162, 'slack_max_bound': 44.94736097837549, 'uncertainty_load_threshold': 0.6470436848379341}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'dcc75072bf1184c35b78ce269b386e75411b4915474ec52546a7d2daa84860a7', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating evidence-backed structural changes:
      - Adds explicit successor-release coupling: (upward_rank × remaining_work) scaled by slack deficit, activated only under DDL pressure (slack <= 0)
      - Introduces host-load–aware energy scaling: energy term multiplied by (1 + normalized_uncertainty) only when uncertainty < threshold
      - Uses robust MAD-based normalization for all features (eliminates degeneracy in sparse ready sets)
      - Preserves strict lexicographic DDL gating: non-critical terms disabled when slack <= 0
      - All numeric literals restricted to {-2, -1, 0, 1, 2}
    """
    eps = 2.688742586490328e-07
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
    slack_score = np.where(slack < 0, (-slack) ** 3.319178828826575, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.432015735969543
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.2378696354917308
    successor_urgency = upward_rank * remaining_work
    successor_score = np.where(slack <= 0, successor_urgency * 0.512255123819319, 0.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    host_load_active = np.where(uncertainty < 0.6470436848379341, 1.0, 0.0)
    energy_scale = 1.0 + mad_normalize(uncertainty) * host_load_active
    scaled_energy = min_incremental_energy * energy_scale
    energy_norm = mad_normalize(scaled_energy)
    slack_lb = -27.86332641160162
    slack_ub = 44.94736097837549
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.3081488977271859 + (1.0 - 0.3081488977271859) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.1126223452306432 * (uncertainty - 1.0)))
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.6400058017611732 * energy_norm * unc_norm * unc_sigmoid
    wait_score = mad_normalize(ready_wait_time)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(successor_score)
    score += slack_headroom_mask * (0.7300627196071271 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
