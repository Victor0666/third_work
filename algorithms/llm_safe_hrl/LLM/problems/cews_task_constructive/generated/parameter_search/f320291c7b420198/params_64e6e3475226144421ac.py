import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's smooth sigmoid urgency and arctan wait saturation
    with Parent 1's work-scaled bottleneck coupling and eliminates redundant parameters.
    
    Key structural improvements:
      - Replaces linear bottleneck_pressure = duration * (remaining_work + upward_rank)
        with power-law scaled version: duration * upward_rank * (remaining_work)^exponent
        to decouple descendant workload magnitude from rank while preserving critical path signal.
      - Retains smooth sigmoid urgency (Parent 2) but tightens clipping and steepness for sharper
        deadline transitions near zero slack — improves hard-DDL adherence.
      - Keeps arctan-saturated wait term (Parent 2) for bounded fairness under long queues.
      - Uses percentile-based critical gating (Parent 2) but applies it to norm_rank instead of raw rank
        for scale-invariant activation.
      - Removes all inactive or redundant parameters (e.g., ddl_risk_threshold, wait_clip_threshold)
        and consolidates logic into 12 well-justified tunables.
      - All operations are finite, deterministic, and protect against NaN/inf/zero.
    """
    eps = 1.2745970344562589e-05
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
        q_low = np.percentile(x, 29.469476692217015)
        q_high = np.percentile(x, 70.81759545602281)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    slack_centered = slack - -0.4890704536846242
    sigmoid_input = np.clip(-4.209925056364574 * slack_centered, -5.118554462918991, 5.118554462918991)
    urgency_gate = 2.0 / (1.0 + np.exp(sigmoid_input)) - 1.0
    norm_urgency = iqr_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    work_scaled = np.power(np.maximum(remaining_work, eps), 0.6078680563884562)
    bottleneck_pressure = duration * (upward_rank + eps) * work_scaled
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    norm_rank = iqr_normalize(upward_rank)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_norm_ranks = np.sort(norm_rank)
        rank_idx = np.searchsorted(sorted_norm_ranks, norm_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.6833894738856009, 1.0, 0.0)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (1.39633633764634 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = iqr_normalize(wait_saturation)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < np.median(slack)) & (uncertainty > 0.7649321936158502 * max_uncertainty)).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    score = norm_urgency + 1.3685085710247922 * norm_bottleneck + 0.5856627148842366 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_active * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
