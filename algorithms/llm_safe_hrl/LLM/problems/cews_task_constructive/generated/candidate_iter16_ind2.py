import numpy as np
RULE_METADATA = {'structure_hash': '5a46049caee962602ee5bcf8af1f6e41ed9d90b67cda9ba586c7e18a36fbe3c6', 'parameter_schema_hash': '2dcdd1c023f9bdfc65f8dbaca775daaa183beb27961704e67897d343a9ee5092', 'best_parameter_hash': '9b07a786fde80e18cfcdbdc237812bda4e453db2ea7345f73f8ddb6420914859', 'best_parameters': {'epsilon': 1.4550518090964598e-05, 'iqr_low_percentile': 25.515490169798213, 'iqr_high_percentile': 78.52504523335492, 'upward_rank_remaining_work_coupling': 0.5093496519647656, 'slack_risk_penalty_weight': 2.7820588794783316, 'uncertainty_slack_coupling': 0.04427318495058535, 'energy_normalization_offset': 0.130164726764065, 'ddl_protection_gate_threshold': 0.7423374501509815}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'b53c4a1093c2874a20ad590ffb01a5183d0ccc8674a66ac2ca1b2d80e7b5e8c2', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

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
    eps = 1.4550518090964598e-05
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
        q_low = np.percentile(x, 25.515490169798213)
        q_high = np.percentile(x, 78.52504523335492)
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
    uncertainty_gate = (uncertainty > 0.7423374501509815 * max_uncertainty).astype(float)
    joint_risk = neg_slack * (uncertainty + eps) * uncertainty_gate
    norm_joint_risk = adaptive_normalize(joint_risk)
    norm_energy = adaptive_normalize(min_incremental_energy)
    norm_energy_shifted = norm_energy + 0.130164726764065
    norm_wait = adaptive_normalize(ready_wait_time)
    inv_wait = 1.0 - norm_wait
    score = 2.7820588794783316 * norm_neg_slack + 0.04427318495058535 * norm_joint_risk + 0.5093496519647656 * norm_bottleneck + norm_energy_shifted + inv_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
