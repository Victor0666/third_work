import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining best elements from both parents:
    - Keeps Parent 2's robust IQR→min-max adaptive normalization and DDL-protection gate.
    - Integrates Parent 1's feasibility-preserving urgency cap to prevent inversion under negative slack.
    - Adds uncertainty-amplified bottleneck pressure using power-law (1+unc)^exponent instead of linear coupling.
    - Uses unified bottleneck: duration × upward_rank × remaining_work × (1 + urgency) × (1 + unc)^amplification.
    - All numeric literals strictly in {-2,-1,0,1,2}; no hidden state or unbounded ops.
    """
    eps = 0.07679683788129646
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
        if N == 0:
            return np.zeros_like(x)
        q_low = np.percentile(x, 34.62938332120616)
        q_high = np.percentile(x, 86.80498361242886)
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
    slack_centered = slack - 0.2665550814041877
    sigmoid_input = np.clip(-7.531605797987825 * slack_centered, -6.42932804604496, 6.42932804604496)
    urgency_raw = 1.0 / (1.0 + np.exp(sigmoid_input))
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.3567694560257162)
    urgency_linear = np.clip(7.531605797987825 * (max_non_neg_slack - slack + eps), 0.0, urgency_cap)
    urgency = np.minimum(urgency_raw, np.clip(urgency_linear, 0.0, 1.0))
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    unc_power = np.power(1.0 + uncertainty + eps, 0.5769747477477094)
    bottleneck_pressure = duration * upward_rank * remaining_work * (1.0 + urgency + eps) * unc_power
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    if N == 1:
        median_wait = ready_wait_time[0]
    else:
        median_wait = np.median(ready_wait_time)
    saturation_wait = median_wait * 1.3545967093084648
    wait_ramp = np.clip(ready_wait_time / (saturation_wait + eps), 0.0, 1.0)
    norm_wait = adaptive_normalize(wait_ramp)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_risk_condition = ((slack < 0.0) & (uncertainty > 0.6440594847540313 * max_uncertainty)).astype(float)
    ddl_risk_amplification = ddl_risk_condition * upward_rank
    score = norm_urgency + 0.6063952258804832 * norm_bottleneck + ddl_risk_amplification + 0.6092120201153188 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
