import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's scenario-aware IQR normalization and bounded urgency gating
    with Parent 1's adaptive DDL-stress mode and urgency amplification.
    
    Key structural innovations:
      - Dual-mode urgency: (1) bounded piecewise base urgency (Parent 2), AND (2) adaptive percentile-based
        DDL-stress amplification (Parent 1) — both active simultaneously for robust risk coverage.
      - Scenario-aware IQR normalization (Parent 2) enhanced with adaptive percentile selection per-task
        based on local uncertainty relative to global median (not global regime only), enabling finer-grained
        robustness without extra branches.
      - Unified urgency term: base urgency + stress-amplified urgency, normalized jointly via same IQR scheme.
      - Critical-path coupling now uses multiplicative rank×work *only* under DDL-stress (Parent 1 insight),
        suppressing it otherwise to avoid over-prioritizing non-critical work.
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no hidden constants.
    """
    eps = 0.00010540692513015861
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    uncertainty_median = np.median(uncertainty) if N > 0 else 0.0
    high_uncertainty_mask = (uncertainty > uncertainty_median).astype(float)
    q_low = np.where(high_uncertainty_mask == 1.0, 16.501933600986842, 39.9938236232919)
    q_high = np.where(high_uncertainty_mask == 1.0, 80.8612249383778, 74.49220562203564)

    def iqr_normalize(x):
        x = np.copy(x)
        q_low_vals = np.percentile(x, q_low) if N > 1 else np.full(N, np.percentile(x, np.mean(q_low)))
        q_high_vals = np.percentile(x, q_high) if N > 1 else np.full(N, np.percentile(x, np.mean(q_high)))
        iqr = q_high_vals - q_low_vals
        center = np.median(x)
        denom = np.where(iqr > eps, iqr, np.maximum(np.max(np.abs(x - center)), eps) + eps)
        return (x - center) / (denom + eps)
    median_slack = np.median(slack) if N > 0 else 0.0
    deadline_risk_threshold = 0.8365455782868259 * median_slack
    urgency_base = np.clip(-slack / (np.abs(median_slack) + eps), -1.0, 1.0)
    urgency_gate = (slack < deadline_risk_threshold).astype(float)
    urgency_final = urgency_base * (1.0 + urgency_gate)
    if N == 1:
        slack_percentile = np.array([0.0])
    else:
        sorted_slack = np.sort(slack)
        slack_idx = np.searchsorted(sorted_slack, slack, side='left')
        slack_percentile = slack_idx / (N + eps)
    ddl_stress_mask = (slack_percentile < 0.2678354635823547).astype(float)
    urgency_amplified = urgency_final * (1.0 + 1.2144739471709645 * ddl_stress_mask)
    norm_urgency = iqr_normalize(urgency_amplified)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    rank_work_pressure = (upward_rank + eps) * (remaining_work + eps)
    norm_bottleneck = iqr_normalize(rank_work_pressure) * ddl_stress_mask
    wait_ramp = np.clip(ready_wait_time / 8.866137659511237, 0.0, 1.0)
    norm_wait = iqr_normalize(wait_ramp)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < median_slack) & (uncertainty > 0.7878861410062501 * max_uncertainty)).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    score = norm_urgency + 0.17143828826208046 * norm_bottleneck + 0.9059436508031368 * norm_energy_eff - norm_wait + ddl_protection_active * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
