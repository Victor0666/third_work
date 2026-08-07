import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining best practices from both parents:
    - Retains Parent 2's robust bounded sigmoid wait saturation and direct urgency-upward_rank coupling.
    - Integrates Parent 1's adaptive urgency steepness modulation *only* when uncertainty is high relative to global max,
      but now gated by joint DDL pressure (slack < median_slack AND uncertainty > threshold) to avoid over-amplification.
    - Introduces novel uncertainty-powered bottleneck amplification: exponentiates uncertainty influence in bottleneck term
      to sharpen release pressure on high-risk blocking tasks.
    - Uses adaptive IQR normalization with fallback to min-max (Parent 2) for stability across feature distributions.
    - Removes inactive critical-path gating entirely; replaces with urgency-coupled upward_rank scaling.
    - All operations are finite, deterministic, and protect against NaN/inf/zero.
    """
    eps = 1.8041823389299114e-05
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
        q_low = np.percentile(x, 35.51441025026865)
        q_high = np.percentile(x, 70.68680199511999)
        iqr = q_high - q_low
        center = np.median(x)
        if iqr > eps:
            denom = iqr
        else:
            denom = np.max(x) - np.min(x)
            if denom < eps:
                denom = eps
        return (x - center) / (denom + eps)
    slack_centered = slack - 0.9880108707385222
    sigmoid_input = np.clip(-7.755357115816755 * slack_centered, -17.19356833170445, 17.19356833170445)
    urgency_gate = 1.0 / (1.0 + np.exp(sigmoid_input))
    median_slack = np.median(slack) if N > 0 else 0.0
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    joint_ddl_pressure = ((slack < median_slack) & (uncertainty > 0.6522920134776578 * max_uncertainty)).astype(float)
    modulated_steepness = 7.755357115816755 * (1.0 + 0.9274160140430914 * joint_ddl_pressure)
    modulated_sigmoid_input = np.clip(-modulated_steepness * slack_centered, -17.19356833170445, 17.19356833170445)
    modulated_urgency = 1.0 / (1.0 + np.exp(modulated_sigmoid_input))
    norm_urgency = adaptive_normalize(modulated_urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + modulated_urgency + eps)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + uncertainty, 0.9534153664465924)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (3.7114782988840442 + eps)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_scaled))
    norm_wait = adaptive_normalize(wait_saturation)
    ddl_risk_amplifier = joint_ddl_pressure * adaptive_normalize(uncertainty)
    score = norm_urgency + 0.002983102651492875 * norm_bottleneck + 1.035928112570635 * norm_energy_eff - norm_wait + ddl_risk_amplifier
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
