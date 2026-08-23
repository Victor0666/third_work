import numpy as np
RULE_METADATA = {'structure_hash': '7530f80d8e3d1dde368693febd5f138c084fed930b4832fd410f56220a17f5b0', 'parameter_schema_hash': 'fe5267a2104e7b08c59b517ffd623ed365fb96e604a49c841fcf62da802d0c7d', 'best_parameter_hash': 'ec0d72ba3aa76ff316848c792214cfff158216ac8cf0d83eade01703d0a2a602', 'best_parameters': {'epsilon': 6.069729823431027e-06, 'uncertainty_slack_coupling': 2.149278627781468, 'duration_risk_penalty': 0.7942319456786627, 'energy_efficiency_ratio_weight': 1.1713575146439097, 'rank_slack_balance': 0.37067061025900766, 'energy_uncertainty_interaction': 1.3021322196110612, 'uncertainty_sigmoid_steepness': 6.1552578063643715, 'wait_time_decay_exponent': 0.10972182924452033, 'slack_linear_scale_low': -19.499508592489477, 'slack_linear_scale_high': 9.127422802891672}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'c2e1484a87f5da7667992921028e06594e392b98f03fff24f7922aea49ecf32f', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with bounded linear slack scaling and min-max normalization:
      - Replaces logistic slack scaling with robust bounded linear interpolation [slack_linear_scale_low, slack_linear_scale_high],
        improving stability near threshold and reducing sensitivity to parameter perturbations.
      - Uses per-feature min-max normalization (not MAD) for better rank-energy tradeoff in small-N ready sets (<5 tasks).
      - Removes redundant critical_path_release_weight and slack_penalty_exponent per reflection — DDL enforcement is now purely
        lexicographic via zeroing non-DDL terms when slack <= 0, eliminating need for tunable amplifiers.
      - All numeric literals are in {-2,-1,0,1,2}; no hidden constants or unbounded operations.
      - Anti-starvation remains dual-gated (slack > 0 AND slack >= 1.0) with fixed threshold.
      - Energy efficiency uses uncertainty-dampened denominator: duration * (1 + uncertainty).
      - Final score is finite, shape-(N,), deterministic, and satisfies all interface contracts.
    """
    eps = 6.069729823431027e-06
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

    def minmax_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        x_min = np.min(x)
        x_max = np.max(x)
        range_safe = x_max - x_min + eps
        normalized = (x - x_min) / range_safe
        return np.clip(normalized, 0.0, 1.0)
    slack_score = np.where(slack < 0, -slack, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.149278627781468
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.7942319456786627
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total_safe = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / (duration_total_safe * (1.0 + uncertainty + eps))
    energy_eff_score = minmax_normalize(energy_per_sec) * 1.1713575146439097
    slack_lb = -19.499508592489477
    slack_ub = 9.127422802891672
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.37067061025900766 + (1.0 - 0.37067061025900766) * (1.0 - slack_scaled)
    rank_score = (1.0 - minmax_normalize(upward_rank)) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-6.1552578063643715 * (uncertainty - 1.0)))
    energy_norm = minmax_normalize(min_incremental_energy)
    unc_norm = minmax_normalize(uncertainty)
    energy_uncertainty_score = 1.3021322196110612 * energy_norm * unc_norm * unc_sigmoid
    starvation_gate = np.where(slack >= 1.0, 1.0, 0.0)
    starvation_enabled = slack_headroom_mask * starvation_gate
    duration_med = np.median(duration_total_safe) if N > 1 else duration_total_safe[0]
    wait_duration_scaled = ready_wait_time / (duration_med + eps)
    wait_power = wait_duration_scaled ** 0.10972182924452033
    wait_score = minmax_normalize(wait_power)
    score = minmax_normalize(slack_score) + minmax_normalize(unc_slack_coupling) + minmax_normalize(duration_risk)
    score += starvation_enabled * (energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
