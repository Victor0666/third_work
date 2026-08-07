import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three structural improvements:
      - Hard DDL protection gate: zero-weights non-critical-path tasks (upward_rank < median) when slack < threshold, preserving dominant neg_slack while preventing misprioritization under deadline pressure.
      - Conditional multiplicative activation: successor-release and bottleneck terms only activate when slack < median_slack, avoiding spurious pressure in slack-rich regimes.
      - Simplified adaptive normalization: uses fixed-range fallback only (no uncertainty std dependency), improving generalization and eliminating fragile risk-context coupling.
    """
    eps = 0.0001942024131817668
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
        center = np.median(x) if N > 0 else 0.0
        range_val = np.max(x) - np.min(x) if N > 0 else eps
        denom = np.maximum(range_val, eps)
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.7021457477725144)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = adaptive_normalize(urgency_linear)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * remaining_work * (1.0 + urgency_linear + eps)
    unc_normalized = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + unc_normalized, 0.5021751052256571)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    slack_abs = np.abs(slack) + eps
    release_decay = np.power(1.0 + slack_abs, -0.7768564265374448)
    successor_release_pressure = upward_rank * remaining_work * release_decay
    norm_successor_release = adaptive_normalize(successor_release_pressure)
    wait_scaled = ready_wait_time / (5.290756551823499 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = adaptive_normalize(wait_saturation)
    cp_coupling = upward_rank * remaining_work
    norm_cp_coupling = adaptive_normalize(cp_coupling)
    median_upward_rank = np.median(upward_rank) if N > 0 else 0.0
    is_critical_path = (upward_rank >= median_upward_rank).astype(float)
    ddl_gate = (slack >= -1.080569689657891).astype(float)
    critical_path_gate = np.where((slack < -1.080569689657891) & (upward_rank < median_upward_rank), 0.0, 1.0)
    pressure_activation = (slack < median_slack).astype(float)
    score = neg_slack + norm_urgency + 0.8280717806682221 * pressure_activation * norm_successor_release + 0.8280717806682221 * pressure_activation * norm_bottleneck + 1.2514081644974218 * norm_energy_eff - norm_wait + 1.9532336084306279 * norm_cp_coupling + (1.0 - critical_path_gate) * 1000.0
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
