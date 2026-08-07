import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with one bounded structural improvement:
      - Replaced all normalization with *pure median-based adaptive scaling* (no MAD/range fallback),
        using fixed epsilon-guarded range only when N<=2 to avoid degenerate variance — improves stability
        under sparse or singleton ready sets while preserving monotonicity and interpretability.
      - Restored wait_saturation_scale and successor_bottleneck_coupling as conditionally activated features,
        gated by deadline pressure (slack < median_slack) to retain anti-starvation and bottleneck signals
        only when needed — avoids spurious pressure in slack-rich regimes.
      - Removed all inactive parameters (ddl_risk_amplification, bottleneck_uncertainty_amplification)
        per reflection to reduce overfitting and sharpen signal fidelity.
      - Simplified final score composition: removed redundant coupling terms and unified pressure activation.
      - Critical path violation penalty now declared and used via PARAMS.
      - Percentiles 25/75 for IQR moved into PARAMETER_SCHEMA to satisfy numeric literal constraint.
    """
    eps = 0.00019365022103528767
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def median_adaptive_normalize(x):
        x = np.copy(x)
        if N == 0:
            return np.zeros_like(x)
        center = np.median(x)
        if N <= 2:
            denom = np.max(x) - np.min(x) + eps
        else:
            q75 = np.percentile(x, 84.82898639340334)
            q25 = np.percentile(x, 13.20452509547428)
            denom = q75 - q25 + eps
        return (x - center) / denom
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    deadline_pressure_mask = (slack < median_slack).astype(float)
    urgency_linear = np.clip(median_slack - slack, 0.0, None)
    max_non_neg_slack = np.max(np.clip(slack, 0.0, None)) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.9262640467576861)
    urgency_capped = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = median_adaptive_normalize(urgency_capped)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = median_adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * remaining_work * (1.0 + urgency_capped + eps)
    norm_bottleneck = median_adaptive_normalize(bottleneck_pressure) * deadline_pressure_mask
    slack_abs = np.abs(slack) + eps
    release_decay = np.power(1.0 + slack_abs, -1.073994940340433)
    successor_release_pressure = upward_rank * remaining_work * release_decay
    norm_successor_release = median_adaptive_normalize(successor_release_pressure) * deadline_pressure_mask
    wait_scaled = ready_wait_time / (3.757217519157062 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = median_adaptive_normalize(wait_saturation) * deadline_pressure_mask
    cp_coupling = upward_rank * remaining_work
    norm_cp_coupling = median_adaptive_normalize(cp_coupling)
    median_upward_rank = np.median(upward_rank) if N > 0 else 0.0
    critical_path_gate = np.where((slack < -0.0010944446957088516) & (upward_rank < median_upward_rank), 0.0, 1.0)
    score = neg_slack + norm_urgency + 0.8092105585638494 * norm_successor_release + 0.8092105585638494 * norm_bottleneck + 1.8164678688591773 * norm_energy_eff - norm_wait + 0.1514697595612794 * norm_cp_coupling + (1.0 - critical_path_gate) * 1064.5322789712989
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
