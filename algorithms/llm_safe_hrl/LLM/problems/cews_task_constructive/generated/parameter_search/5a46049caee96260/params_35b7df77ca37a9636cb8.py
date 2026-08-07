import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
    - Direct clipped negative slack penalty (not sigmoid) for sharper, more interpretable DDL-risk signal
    - Explicit upward_rank × remaining_work interaction to capture successor-release bottleneck pressure
    - Joint uncertainty-slack coupling to amplify priority when both deadline risk and execution uncertainty are high
    - Energy term with additive offset to avoid suppression in low-energy regimes
    - All normalizations use adaptive IQR→min-max fallback; no unbounded ops or hidden state.
    - Removed wait ramp and urgency sigmoid: replaced by linear anti-starvation via ready_wait_time scaling
    - ddl_protection_gate_threshold now used in joint risk term to gate amplification only under high uncertainty
    """
    eps = 3.7244299148493186e-05
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
        q_low = np.percentile(x, 32.35039474273252)
        q_high = np.percentile(x, 89.99842994675498)
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
    neg_slack = np.clip(-slack, 0.0, None)
    norm_neg_slack = adaptive_normalize(neg_slack)
    bottleneck_term = upward_rank * remaining_work
    norm_bottleneck = adaptive_normalize(bottleneck_term)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    uncertainty_gate = (uncertainty > 0.7011396190284656 * max_uncertainty).astype(float)
    joint_risk = neg_slack * (uncertainty + eps) * uncertainty_gate
    norm_joint_risk = adaptive_normalize(joint_risk)
    norm_energy = adaptive_normalize(min_incremental_energy)
    norm_energy_shifted = norm_energy + 0.1937691032142754
    norm_wait = adaptive_normalize(ready_wait_time)
    inv_wait = 1.0 - norm_wait
    score = 0.12306931617929745 * norm_neg_slack + 0.12126974972394185 * norm_joint_risk + 0.8608066739003856 * norm_bottleneck + norm_energy_shifted + inv_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
