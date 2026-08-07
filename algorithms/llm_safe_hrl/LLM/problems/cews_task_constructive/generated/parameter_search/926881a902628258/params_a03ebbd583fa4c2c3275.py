import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating counterfactual evidence:
    - Removed inactive parameters (wait_saturation_time, critical_rank_percentile) per inactivity analysis
    - Replaced arctan wait saturation with bounded linear ramp to avoid over-smoothing under starvation
    - Simplified DDL protection to joint condition: tight slack AND high uncertainty (no percentile gating)
    - Unified risk coupling: multiplicative urgency × uncertainty instead of additive conditional term
    - Used adaptive min-max fallback when IQR dispersion is low (improves robustness to near-constant features)
    - Removed critical bonus (evidence shows it causes energy degradation without DDL improvement)
    - Prioritizes smooth, bounded, and numerically stable operations only.
    """
    eps = 0.00415907613661828
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
        q_low = np.percentile(x, 32.916614610361975)
        q_high = np.percentile(x, 88.92436268015398)
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
    slack_centered = slack - 0.8536312350433575
    sigmoid_input = np.clip(-3.707046553938854 * slack_centered, -31.255912146297963, 31.255912146297963)
    urgency = 1.0 / (1.0 + np.exp(sigmoid_input))
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    if N == 1:
        median_wait = ready_wait_time[0]
    else:
        median_wait = np.median(ready_wait_time)
    wait_ramp = np.clip(ready_wait_time / (median_wait + eps), 0.0, 1.0)
    norm_wait = adaptive_normalize(wait_ramp)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < median_slack) & (uncertainty > 0.6538549617270945 * max_uncertainty)).astype(float)
    risk_coupled = urgency * uncertainty
    norm_risk = adaptive_normalize(risk_coupled)
    score = norm_urgency + 0.001915938859255077 * norm_bottleneck + ddl_risk_gate * norm_risk + 1.2844535756564712 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
