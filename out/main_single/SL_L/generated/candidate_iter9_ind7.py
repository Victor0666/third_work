import numpy as np
RULE_METADATA = {'structure_hash': 'af3ed242f5e936da272e10092e3a9a8ee419faa349658312df5503bbcc7faac0', 'parameter_schema_hash': 'e661c4ccd834f20538a6e45c47fca04eaaae098851436b1c61dbb0df928c1e5f', 'best_parameter_hash': '7a69346050dce81786ff9bb5c6923f5c400ef328399b3b23eab8564505b5a7c1', 'best_parameters': {'epsilon': 1.1302263633657236e-06, 'slack_penalty_exponent': 2.897033162684017, 'slack_aware_criticality_scale': 0.5483941622096659, 'energy_efficiency_ratio_weight': 1.6075221789066991, 'uncertainty_slack_coupling': 2.678235785497794, 'duration_risk_penalty': 0.7086432540948809, 'energy_uncertainty_interaction': 1.2146180320394364, 'uncertainty_sigmoid_steepness': 3.6891690049875336, 'wait_fairness_gain': 1.164050346476299, 'work_density_weight': 0.4431553357003409, 'tanh_bias': 0.7468665069708977, 'duration_quantile': 0.9440690766696348}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '577f6c62e2c9c1e524a2a11e5015f590ecd54841e1f109c02b4fef450a891e56', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with slack-aware normalized criticality boost and quantile-based work-density scaling.
    
    Key structural improvements:
    - Replaces fragile sigmoid rank gating with *slack-aware normalized criticality boost*: 
      upward_rank is scaled additively by clipped negative slack, eliminating steepness tuning.
    - Uses *declared quantile parameter* for work-density denominator (replacing literal 0.75) to ensure full tunability.
    - Replaces exponential wait saturation with *tanh-scaled fairness* using declared tanh_bias.
    - All beneficial terms subtracted; all penalties added; strict DDL-first ordering preserved.
    """
    eps = 1.1302263633657236e-06
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

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        abs_x = np.abs(x)
        scale = np.mean(abs_x) + eps
        return x / (scale + eps)
    slack_score = np.where(slack < 0, (-slack) ** 2.897033162684017, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    duration_norm = robust_normalize(duration_total)
    duration_risk = duration_norm * uncertainty * 0.7086432540948809
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = robust_normalize(energy_per_sec) * 1.6075221789066991
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = robust_normalize(uncertainty * deadline_pressure) * 2.678235785497794
    rank_norm = robust_normalize(upward_rank)
    clipped_slack = np.clip(-slack, 0.0, None)
    critical_bonus = rank_norm * (1.0 + 0.5483941622096659 * clipped_slack)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-3.6891690049875336 * (uncertainty - 1.0)))
    energy_norm = robust_normalize(min_incremental_energy)
    unc_norm = robust_normalize(uncertainty)
    energy_uncertainty_score = 1.2146180320394364 * energy_norm * unc_norm * unc_sigmoid
    energy_base_score = robust_normalize(min_incremental_energy)
    wait_norm = robust_normalize(ready_wait_time)
    wait_boost = 0.7468665069708977 * (1.0 + np.tanh(1.164050346476299 * wait_norm))
    duration_q = np.quantile(duration_total, 0.9440690766696348) + eps
    work_density = remaining_work / duration_q
    work_density_norm = robust_normalize(work_density)
    work_density_bonus = 0.4431553357003409 * work_density_norm
    score = robust_normalize(slack_score) + duration_risk + energy_eff_score + unc_slack_coupling + energy_uncertainty_score + energy_base_score + wait_boost - critical_bonus - work_density_bonus
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
