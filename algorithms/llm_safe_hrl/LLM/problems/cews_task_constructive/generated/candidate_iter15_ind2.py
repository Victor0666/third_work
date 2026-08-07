import numpy as np
RULE_METADATA = {'structure_hash': '3c4f480efa8e5eef88fe3765a3b605c06a44fad4472da0c122907ade8421df2e', 'parameter_schema_hash': '844b5275991335aa85cd1ffa040bf1f7954e6876a62232752d6bf136f61c8371', 'best_parameter_hash': 'd64444d78e982155c843283db17260b03c008dd1bd17a72d612382e5ec4fb711', 'best_parameters': {'epsilon': 2.487793489893154e-06, 'ddl_protection_gate_threshold': 0.9415568245331094, 'energy_duration_ratio_weight': 1.2776685736085884, 'successor_bottleneck_coupling': 0.6143828629024842, 'iqr_low_percentile': 35.6903592721359, 'iqr_high_percentile': 74.31620579070355, 'tanh_urgency_scale': 2.7114091646291345, 'bottleneck_uncertainty_coupling': 0.02643093839381287, 'dynamic_feasibility_threshold_factor': 1.3121324671495864, 'critical_path_interaction_exponent': 1.5235036195004756, 'wait_boost_under_ddl_risk': 0.8490658069866142, 'urgency_clamp_lower': -0.795011149779949}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '9fe01c07688931bcfe6f1ceb43561e4247b43e181293c9f2ef39d63df8f89440', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with tanh-based urgency, additive uncertainty coupling, and clamped DDL-risk sensitivity.
    
    Key structural improvements:
      - Replaces sigmoid urgency with numerically stable tanh: `tanh(scale * slack)` → [-1,1], then clamped to avoid pathological penalties.
      - Converts bottleneck uncertainty coupling from multiplicative to *additive*: `+ coupling * uncertainty`, preventing explosion under high uncertainty.
      - Introduces explicit `urgency_clamp_lower` to ensure tasks with marginally negative slack aren't over-penalized — preserves fairness while maintaining hard-DDL safety.
      - Retains dynamic feasibility gating and exponentiated critical-path under joint DDL-risk.
      - Keeps anti-starvation wait boost activated only when slack < 0.
      - All operations remain finite, deterministic, and numerically safe.
    """
    eps = 2.487793489893154e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def adaptive_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 35.6903592721359)
        q_high = np.percentile(x, 74.31620579070355)
        iqr = q_high - q_low
        center = np.median(x)
        if iqr < eps:
            x_min, x_max = (np.min(x), np.max(x))
            denom = x_max - x_min
            if denom < eps:
                return np.zeros_like(x)
            return (x - x_min) / (denom + eps)
        else:
            denom = iqr
            return (x - center) / (denom + eps)
    tanh_input = 2.7114091646291345 * slack
    urgency = np.tanh(tanh_input)
    urgency = np.clip(urgency, -0.795011149779949, 1.0)
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency + eps) + 0.02643093839381287 * uncertainty * duration
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    if N == 1:
        median_wait = ready_wait_time[0]
    else:
        median_wait = np.median(ready_wait_time)
    wait_ramp = np.clip(ready_wait_time / (median_wait + eps), 0.0, 1.0)
    norm_wait = adaptive_normalize(wait_ramp)
    slack_penalty_mask = (slack < 0.0).astype(float)
    boosted_wait = norm_wait * (1.0 + slack_penalty_mask * 0.8490658069866142)
    slack_iqr = np.percentile(slack, 74.31620579070355) - np.percentile(slack, 35.6903592721359)
    dynamic_feasibility_threshold = np.median(slack) + 1.3121324671495864 * slack_iqr
    feasibility_gate = np.where(slack > dynamic_feasibility_threshold, 0.0, 1.0)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < median_slack) & (uncertainty > 0.9415568245331094 * max_uncertainty)).astype(float)
    critical_path_strength = (upward_rank * remaining_work + eps) ** 1.5235036195004756
    norm_critical_path = adaptive_normalize(critical_path_strength)
    score = norm_urgency + 0.6143828629024842 * ddl_risk_gate * norm_critical_path + feasibility_gate * 1.2776685736085884 * norm_energy_eff - boosted_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
