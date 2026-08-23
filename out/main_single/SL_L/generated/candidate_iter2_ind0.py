import numpy as np
RULE_METADATA = {'structure_hash': 'd3ed9ef2b74aa61c31e5425823eb7e6fdbb22a8180af54a5ebc3462cab2b5235', 'parameter_schema_hash': 'dcabd4b3b458ce9a610e7b2433a8ab7a4d8feabe9f595f2a931dbaf091508137', 'best_parameter_hash': '88238474cfbc48739c597749178d034b24b3663a504035bf8d30395f02f298c1', 'best_parameters': {'epsilon': 1.3295979487002373e-09, 'slack_penalty_exponent': 1.4323055740517714, 'criticality_boost': 2.024325199843874, 'energy_sensitivity': 0.4862475883481875, 'duration_balance': 1.401380846244619, 'wait_decay_rate': 0.1813871548538273, 'uncertainty_slack_coupling': 2.108920195278771, 'rank_energy_interaction': 0.5902852949045985, 'urgency_clip_max': 5.4193874928139065, 'work_density_weight': 0.013411824464523726, 'starvation_uncertainty_mod': 0.6154667062310689}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'fc8c279d6cd1e68425fa32f2688bef69ca853e14a9213396194bfefdac77d6bf', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Hybrid priority rule combining Parent 2's exponential urgency & gated criticality with Parent 1's
    work-density bonus and starvation-uncertainty modulation. Introduces novel starvation modulation:
    wait boost scaled inversely by uncertainty to avoid over-committing to old tasks in volatile environments.
    All normalizations use mean-abs + epsilon for stability; no unbounded ops or loops.
    """
    eps = 1.3295979487002373e-09
    finfo = np.finfo(np.float64)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps)

    def robust_norm(x):
        denom = np.mean(np.abs(x)) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    duration_raw = min_exec_time + min_comm_time
    norm_duration = robust_norm(duration_raw)
    norm_rank = robust_norm(upward_rank)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    work_density = np.divide(remaining_work, duration_raw + eps)
    norm_work_density = robust_norm(work_density)
    slack_sign_mask = (slack < 0).astype(float)
    urgency_base = -norm_slack * slack_sign_mask
    urgency_clipped = np.clip(urgency_base, 0.0, 5.4193874928139065)
    urgency_penalty = np.exp(1.4323055740517714 * urgency_clipped)
    rank_median = np.median(norm_rank)
    critical_mask = (norm_rank >= rank_median).astype(float)
    critical_boost = 2.024325199843874 * norm_rank * critical_mask
    slack_pressure = np.maximum(0.0, -norm_slack)
    uncertainty_coupled_pressure = 2.108920195278771 * slack_pressure * norm_uncert
    wait_boost_base = 1.0 - np.exp(-0.1813871548538273 * norm_wait)
    starvation_mod = np.clip(1.0 - 0.6154667062310689 * norm_uncert, 0.0, 1.0)
    wait_boost = wait_boost_base * starvation_mod
    rank_energy_penalty = 0.5902852949045985 * norm_energy * norm_rank
    work_density_bonus = 0.013411824464523726 * norm_work_density
    score = urgency_penalty + uncertainty_coupled_pressure + rank_energy_penalty + 0.4862475883481875 * norm_energy + 1.401380846244619 * norm_duration - critical_boost - wait_boost - work_density_bonus
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=-finfo.max)
    return score
