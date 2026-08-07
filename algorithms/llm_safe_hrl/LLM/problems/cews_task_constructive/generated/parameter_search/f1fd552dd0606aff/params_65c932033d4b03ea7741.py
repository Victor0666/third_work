import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule synthesizing Parent 2's smooth urgency with Parent 1's robust feasibility gating,
    enhanced by uncertainty-aware amplification and tightened IQR normalization.
    
    Key structural improvements:
      - Replaces binary DDL-critical mask with *adaptive sigmoid urgency* AND *conditional uncertainty amplification*
        — both activated only when slack is tight relative to median AND uncertainty exceeds threshold.
      - Introduces `uncertainty_amplification_weight`: decouples uncertainty's role from urgency signal,
        allowing independent tuning of risk-aware penalty under DDL stress.
      - Tightens IQR percentiles (23/77) for improved outlier resilience in heterogeneous edge-cloud workloads.
      - Uses bottleneck pressure as `duration * (remaining_work + upward_rank + eps)` to avoid zero collapse.
      - Critical-path bonus remains percentile-gated but computed safely for N=1 via rank-indexing.
      - All operations guarded against NaN/inf/zero; deterministic and finite output.
    """
    eps = 0.0002145635704391391
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
        q_low = np.percentile(x, 10.28516875700213)
        q_high = np.percentile(x, 76.04290830748019)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    slack_centered = slack - 0.5319156593467564
    sigmoid_input = np.clip(-6.533130809710712 * slack_centered, -30.843708360029176, 30.843708360029176)
    urgency_gate = 2.0 / (1.0 + np.exp(sigmoid_input)) - 1.0
    norm_urgency = iqr_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    bottleneck_pressure = duration * (remaining_work + upward_rank + eps)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.5757153808885095, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (17.39195804144051 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = iqr_normalize(wait_saturation)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < median_slack) & (uncertainty > 0.655856038445018 * max_uncertainty)).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    score = norm_urgency + 0.7662384974867189 * norm_bottleneck + 0.6838394556351386 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_active * 0.8847680971822101 * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
