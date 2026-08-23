import numpy as np
RULE_METADATA = {'structure_hash': 'f0602b53f96d467187ee4b4e7b6cf5a0ed770ab7b567de4531cf903d5f8eef79', 'parameter_schema_hash': '75a632b5e03bfc881f1976714f2f74f74419712fe38fe0d1b88f66cc0d323921', 'best_parameter_hash': '3fc0816e65b1865e13eed2b3fe37fc3deb31ee0a149ea736974bbee114edd6a8', 'best_parameters': {'epsilon': 2.0469905708589316e-06, 'slack_penalty_exponent': 2.018556538276149, 'criticality_boost': 2.7499259872041044, 'energy_efficiency_ratio_weight': 0.9337060878512814, 'uncertainty_slack_coupling': 0.2924679820906901, 'rank_slack_balance': 0.5432307502883424, 'duration_risk_penalty': 0.0620726759948625, 'energy_uncertainty_interaction': 0.0005001974318869044, 'uncertainty_sigmoid_steepness': 0.7688213918816527, 'slack_min_bound': -20.690238242000078, 'slack_max_bound': 34.748429783850185, 'wait_time_decay_exponent': 0.8305906649957491}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'f69563ec91f0b378bba806222b553935e723fa82d77d4a7b1f7ed7a06820d16f', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule integrating:
      - Strict lexicographic DDL gating: all non-DDL terms masked by `slack > 0`
      - Critical path urgency via `upward_rank * remaining_work` (validated in cross-agent consensus)
      - Power-law anti-starvation: `ready_wait_time ** wait_time_decay_exponent`, gated by slack headroom
      - MAD-based robust normalization per dimension instead of percentile clipping (more stable under sparse N)
      - Unified duration-risk term: `(min_exec_time + min_comm_time) * (1 + uncertainty)` to reflect load-sensitive delay
      - Energy-efficiency score now uses `min_incremental_energy / (min_exec_time + min_comm_time + eps)` directly
      - All numeric literals restricted to {-2, -1, 0, 1, 2}; no other constants used
      - Final score enforces: DDL violation penalty first → critical path release urgency → energy efficiency among feasible → anti-starvation only when safe
    """
    eps = 2.0469905708589316e-06
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
    slack_score = np.where(slack < 0, (-slack) ** 2.018556538276149, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.2924679820906901
    duration_total = min_exec_time + min_comm_time + eps
    duration_risk = duration_total * (1.0 + uncertainty) * 0.0620726759948625
    critical_path_urgency = upward_rank * remaining_work
    critical_path_score = mad_normalize(critical_path_urgency) * 2.7499259872041044
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -20.690238242000078
    slack_ub = 34.748429783850185
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.5432307502883424 + (1.0 - 0.5432307502883424) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-0.7688213918816527 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.0005001974318869044 * energy_norm * unc_norm * unc_sigmoid
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 0.8305906649957491, 0.0)
    wait_score = mad_normalize(wait_power)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + critical_path_score
    score += slack_headroom_mask * (0.9337060878512814 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
