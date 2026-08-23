import numpy as np
RULE_METADATA = {'structure_hash': 'fc97b63be0edb903f50fad473c357faead8bb16504106e8da881feec2963b372', 'parameter_schema_hash': 'c4d4b3977ec3bc4b3eea7ff733403a4719106bc0ba9e42b17b9272629e3ba82b', 'best_parameter_hash': '19164c8e82fdcedda17a79bf16d7ce9164eb8034fb798c6f5cc25f17bc432f15', 'best_parameters': {'epsilon': 1.6205696948878838e-09, 'slack_penalty_exponent': 1.8111192668850438, 'criticality_boost': 3.029773089779951, 'energy_efficiency_ratio_weight': 0.7188808276383969, 'uncertainty_slack_coupling': 1.506464815678275, 'rank_slack_balance': 0.44407697715339806, 'duration_risk_penalty': 0.8679289212764827, 'energy_uncertainty_interaction': 0.7973451282473929, 'uncertainty_sigmoid_steepness': 5.665014427879926, 'slack_min_bound': -3.7956649069714388, 'slack_max_bound': 36.12771505713396, 'wait_time_decay_exponent': 0.9506636934909688}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '9c7d9dbddf0b2b7d51eba0853dcde0028be5014b6e97b3340f8b8dadec895907', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
      - Strict lexicographic DDL gating via hard `slack > 0` mask (simplified from prior threshold)
      - Critical path release prioritization: `upward_rank * remaining_work` interaction scaled by criticality_boost under DDL pressure
      - Power-law anti-starvation: `ready_wait_time ** wait_time_decay_exponent`, gated by slack headroom and normalized robustly
      - MAD-based normalization per dimension instead of percentile (more stable for small N)
      - Explicit slack sign-aware scaling: linear interpolation from 0 to 1 over [slack_min_bound, slack_max_bound] for rank balance
      - All divisions guarded by eps; all inf/nan replaced before computation; no in-place mutation
      - Final score structure: [DDL violation penalty] + [critical path urgency] + [gated non-DDL efficiency & fairness]
    """
    eps = 1.6205696948878838e-09
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
    slack_score = np.where(slack < 0, (-slack) ** 1.8111192668850438, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.506464815678275
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.8679289212764827
    is_ddl_constrained = slack <= 0.0
    successor_release_score = upward_rank * remaining_work * 3.029773089779951
    successor_release_norm = -mad_normalize(successor_release_score)
    successor_release_contribution = np.where(is_ddl_constrained, successor_release_norm, 0.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -3.7956649069714388
    slack_ub = 36.12771505713396
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.44407697715339806 + (1.0 - 0.44407697715339806) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-5.665014427879926 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.7973451282473929 * energy_norm * unc_norm * unc_sigmoid
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 0.9506636934909688, 0.0)
    wait_score = mad_normalize(wait_power)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + successor_release_contribution
    score += slack_headroom_mask * (0.7188808276383969 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
