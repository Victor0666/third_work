import numpy as np
RULE_METADATA = {'structure_hash': 'f4e7b3adc5c653d1d3fa87b5403dce98cf23b5aed475a1d5f66fe113681c2393', 'parameter_schema_hash': '40e7807a397acdb257f63a18a4aa41db0705bc977b5cc855274095b409ac94f2', 'best_parameter_hash': 'fbdbfd62a3f4d322657f4d0e184980a2e623aa0c3d726a5d54397fbf60611622', 'best_parameters': {'epsilon': 3.0868423975439508e-06, 'ddl_risk_gate_threshold': 0.8189480373622161, 'ddl_risk_amplification': 0.6837485854647348, 'critical_path_coupling': 1.7877342522327444, 'energy_normalization_scale': 1.216258072709508, 'uncertainty_slack_coupling': 1.141006741429007, 'wait_decay_exponent': 0.6434957067047193}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '3643e1d73637b23a34d767f928035c416f70dea9c82095283fa5445093557421', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Retains Parent 2's conditional DDL risk gate (slack < threshold * median_slack) for precise hard-constraint enforcement.
      - Integrates Parent 1's bounded power-law wait decay for smoother anti-starvation behavior vs linear boost.
      - Uses robust MAD normalization universally (energy, critical path, bottleneck, wait).
      - Introduces novel *bottleneck-aware duration scaling*: duration-weighted critical path pressure only under DDL risk.
      - Replaces raw neg_slack with power-law DDL violation penalty (Parent 1 style) for stronger dominance near deadline.
      - Uncertainty-slack coupling remains gated by DDL risk (Parent 2), but applied to normalized deficit.
      - All terms additive, deterministic, finite, and use only {-2,-1,0,1,2} literals.
    """
    eps = 3.0868423975439508e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def mad_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        abs_dev = np.abs(x - center)
        mad = np.median(abs_dev) if N > 0 else eps
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = mad if mad > eps else fallback_range
        return (x - center) / (denom + eps)
    median_slack = np.median(slack) if N > 0 else 0.0
    ddl_risk_mask = (slack < 0.8189480373622161 * (median_slack + eps)).astype(float)
    neg_slack = np.clip(-slack, 0.0, None)
    ddl_risk_penalty = np.power(neg_slack + eps, 0.6837485854647348)
    critical_pressure = upward_rank * remaining_work
    critical_pressure = critical_pressure * (1.0 + ddl_risk_mask * (1.7877342522327444 - 1.0))
    norm_critical_pressure = mad_normalize(critical_pressure)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    bottleneck_pressure = duration * critical_pressure * ddl_risk_mask
    norm_bottleneck = mad_normalize(bottleneck_pressure)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy = mad_normalize(energy_per_duration) * 1.216258072709508
    norm_slack_deficit = mad_normalize(neg_slack)
    unc_coupled_deficit = norm_slack_deficit * np.power(1.0 + uncertainty, 1.141006741429007)
    unc_coupled_deficit = unc_coupled_deficit * ddl_risk_mask
    max_wait = np.max(ready_wait_time) if N > 0 else eps
    wait_ratio = np.clip(ready_wait_time / (max_wait + eps), 0.0, 1.0)
    wait_decay = np.power(wait_ratio + eps, 0.6434957067047193)
    wait_priority = 1.0 - wait_decay
    norm_wait = mad_normalize(wait_priority)
    score = ddl_risk_penalty + norm_bottleneck + norm_critical_pressure + norm_energy + mad_normalize(unc_coupled_deficit) - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
