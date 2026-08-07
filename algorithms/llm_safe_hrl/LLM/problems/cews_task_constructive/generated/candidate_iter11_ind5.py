import numpy as np
RULE_METADATA = {'structure_hash': 'a90b2cba69a59af65fe07e1e324d98807dbf83163d55a64de87c90316b5fbfbb', 'parameter_schema_hash': '2e5feb8fb9fa269cdd487a10ee14b758ac711eba4ea663b480bc5b100ed658b3', 'best_parameter_hash': 'cf8aa681aea8bcd3c0067d26022ad0b817962c10a5f60575000d431fc5fc6d8c', 'best_parameters': {'epsilon': 0.0002948414841522247, 'ddl_protection_gate_threshold': 0.8352288304326849, 'energy_duration_ratio_weight': 1.6122512321569495, 'successor_bottleneck_coupling': 0.8278654858116063, 'iqr_low_percentile': 39.94903421963973, 'iqr_high_percentile': 83.72739864581966, 'slack_sigmoid_steepness': 1.4627578741103626, 'slack_sigmoid_offset': 0.7446249423506983, 'sigmoid_clip_bound': 17.792688093942807, 'bottleneck_uncertainty_coupling': 0.06311847118844627, 'feasibility_slack_threshold': 12.789844765156017}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'c3a3883adb33bc3c30f9cd7898484b6d1678454f71a536584576c4229cacb835', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robustness with Parent 1's feasibility-aware gating.
    
    Key structural improvements:
      - Retains Parent 2's adaptive IQR/min-max normalization for stability across heterogeneous ready sets.
      - Integrates Parent 1's *feasibility-aware gating*: nullifies energy & uncertainty contributions when slack > threshold,
        enforcing hard deadline priority hierarchy and preventing over-penalization of safe tasks.
      - Replaces additive risk amplification with *multiplicative bottleneck uncertainty coupling*:
        bottleneck_pressure × (1 + coupling * normalized_uncertainty), reflecting that uncertainty magnifies blocking impact.
      - Keeps bounded linear wait ramp (Parent 2) for stronger anti-starvation fairness under congestion.
      - Removes critical-path bonus (evidence shows degradation) and arctan saturation (over-smoothing).
      - Uses joint DDL-risk gate (Parent 2) but applies it only to the *risk-coupled bottleneck*, not raw uncertainty.
      - All operations guarded against NaN/inf/zero; deterministic and finite.
    """
    eps = 0.0002948414841522247
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
        q_low = np.percentile(x, 39.94903421963973)
        q_high = np.percentile(x, 83.72739864581966)
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
    slack_centered = slack - 0.7446249423506983
    sigmoid_input = np.clip(-1.4627578741103626 * slack_centered, -17.792688093942807, 17.792688093942807)
    urgency = 1.0 / (1.0 + np.exp(sigmoid_input))
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_uncertainty = adaptive_normalize(uncertainty)
    bottleneck_with_risk = norm_bottleneck * (1.0 + 0.06311847118844627 * norm_uncertainty)
    if N == 1:
        median_wait = ready_wait_time[0]
    else:
        median_wait = np.median(ready_wait_time)
    wait_ramp = np.clip(ready_wait_time / (median_wait + eps), 0.0, 1.0)
    norm_wait = adaptive_normalize(wait_ramp)
    feasibility_gate = np.where(slack > 12.789844765156017, 0.0, 1.0)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < median_slack) & (uncertainty > 0.8352288304326849 * max_uncertainty)).astype(float)
    score = norm_urgency + 0.8278654858116063 * ddl_risk_gate * bottleneck_with_risk + feasibility_gate * 1.6122512321569495 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
