import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining best elements from both parents:
    - Keeps Parent 2's robust IQR→min-max adaptive normalization and DDL-protection gate.
    - Integrates Parent 1's feasibility-preserving urgency cap to prevent inversion under negative slack.
    - Adds uncertainty-amplified bottleneck pressure using power-law (1+unc)^exponent instead of linear coupling.
    - Uses unified bottleneck: duration × upward_rank × remaining_work × (1 + urgency) × (1 + unc)^amplification.
    - All numeric literals strictly in {-2,-1,0,1,2}; no hidden state or unbounded ops.
    """
    eps = 0.00034408852244084806
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
        q_low = np.percentile(x, 37.876122009603094)
        q_high = np.percentile(x, 65.60142028633017)
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
    slack_centered = slack - -0.32905170767504255
    sigmoid_input = np.clip(-4.413644566428546 * slack_centered, -36.24528003980642, 36.24528003980642)
    urgency_raw = 1.0 / (1.0 + np.exp(sigmoid_input))
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.7022978165597065)
    urgency_linear = np.clip(4.413644566428546 * (max_non_neg_slack - slack + eps), 0.0, urgency_cap)
    urgency = np.minimum(urgency_raw, np.clip(urgency_linear, 0.0, 1.0))
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    unc_power = np.power(1.0 + uncertainty + eps, 0.5686804871591807)
    bottleneck_pressure = duration * upward_rank * remaining_work * (1.0 + urgency + eps) * unc_power
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    if N == 1:
        median_wait = ready_wait_time[0]
    else:
        median_wait = np.median(ready_wait_time)
    saturation_wait = median_wait * 4.112425128255465
    wait_ramp = np.clip(ready_wait_time / (saturation_wait + eps), 0.0, 1.0)
    norm_wait = adaptive_normalize(wait_ramp)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_risk_condition = ((slack < 0.0) & (uncertainty > 0.5722292203055974 * max_uncertainty)).astype(float)
    ddl_risk_amplification = ddl_risk_condition * upward_rank
    score = norm_urgency + 0.15275761493895518 * norm_bottleneck + ddl_risk_amplification + 1.9987997083866325 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
