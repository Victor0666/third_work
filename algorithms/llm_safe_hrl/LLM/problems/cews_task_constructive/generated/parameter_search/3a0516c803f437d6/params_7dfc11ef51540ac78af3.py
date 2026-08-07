import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust urgency & host-aware energy with Parent 1's hard deadline gating.
    
    Key structural innovations:
      - Dual-mode urgency: linear urgency gate (Parent 2) *plus* hard feasibility guard (Parent 1) — 
        when slack < ddl_feasibility_threshold, urgency is amplified and all non-urgency terms suppressed.
      - Host-load–aware energy normalization preserved from Parent 2, now gated by feasibility mode to avoid over-penalizing under stress.
      - Critical-path bonus retained but computed via safe percentile ranking even for N=1.
      - IQR percentiles tightened to (20/80) for stronger outlier resilience.
      - Wait saturation uses arctan (Parent 2), but fairness term is *suppressed* under hard DDL mode to prioritize deadline compliance.
      - All operations guarded against NaN/inf/zero; deterministic and finite output.
    """
    eps = 0.0009240653569889313
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def iqr_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 27.241937351873005)
        q_high = np.percentile(x, 89.98625323702302)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    median_abs_slack = np.maximum(np.abs(np.median(slack)), eps)
    urgency_gate = np.clip(slack / (median_abs_slack + eps), -1.0, 1.0)
    urgency_gate = -urgency_gate
    ddl_critical_mask = (slack < 2.6793133481071756).astype(float)
    urgency_amplified = urgency_gate * (1.0 + 4.164041874290069 * ddl_critical_mask)
    norm_urgency = iqr_normalize(urgency_amplified)
    host_load_adjusted_energy = min_incremental_energy * (1.0 + 0.3899276250003154 * uncertainty)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = host_load_adjusted_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration) * (1.0 - ddl_critical_mask)
    bottleneck_pressure = duration * (remaining_work + upward_rank + eps)
    norm_bottleneck = iqr_normalize(bottleneck_pressure) * (1.0 - ddl_critical_mask)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.6759720742766719, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank * (1.0 - ddl_critical_mask)
    wait_scaled = ready_wait_time / (12.631618143855212 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = iqr_normalize(wait_saturation) * (1.0 - ddl_critical_mask)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_mask = ((slack < median_slack) & (uncertainty > 0.8033004373306538 * max_uncertainty)).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    risk_amplification_mask = ((slack < 0) & (uncertainty > 0.8033004373306538 * max_uncertainty)).astype(float)
    risk_amplified_urgency = norm_urgency * (1.0 + 0.4845280470011891 * norm_uncertainty * risk_amplification_mask)
    score = risk_amplified_urgency + 0.6967465326934932 * norm_bottleneck + 0.7147545339190171 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_mask * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
