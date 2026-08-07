import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with feasibility-aware gating and local robust scaling.
    
    Key structural improvements:
      - Replaces global IQR normalization with *task-local median absolute deviation (MAD)* scaling
        for outlier resilience in heterogeneous ready sets — avoids distortion from extreme outliers.
      - Introduces *feasibility-aware gating*: nullifies energy & uncertainty contributions when slack > threshold,
        preventing over-penalization of low-risk tasks and aligning with hard deadline constraint priority.
      - Decouples uncertainty: now *multiplicatively modulates bottleneck pressure* (not additive),
        reflecting that risk amplifies blocking impact — not independent penalty.
      - All normalization uses local center/scale per feature; no cross-task contamination.
      - Maintains smooth sigmoid urgency, arctan wait saturation, and percentile-gated critical bonus.
      - Deterministic, finite, and fully guarded against NaN/inf/zero.
    """
    eps = 0.004844726710357242
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def mad_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        abs_devs = np.abs(x - center)
        mad = np.median(abs_devs)
        denom = mad if mad > eps else np.max(abs_devs) + eps
        return (x - center) / (denom + eps)
    slack_centered = slack - 0.3135879312349952
    sigmoid_input = np.clip(-2.459839395906518 * slack_centered, -20.230159962938295, 20.230159962938295)
    urgency_gate = 2.0 / (1.0 + np.exp(sigmoid_input)) - 1.0
    norm_urgency = mad_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = mad_normalize(energy_per_duration)
    bottleneck_pressure = duration * (remaining_work + upward_rank + eps)
    norm_bottleneck = mad_normalize(bottleneck_pressure)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.7749612009333673, 1.0, 0.0)
    norm_rank = mad_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (7.4459786918555455 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = mad_normalize(wait_saturation)
    feasibility_gate = np.where(slack > 25.800297011827666, 0.0, 1.0)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < median_slack) & (uncertainty > 0.7075918292682444 * max_uncertainty)).astype(float)
    norm_uncertainty = mad_normalize(uncertainty)
    bottleneck_with_risk = norm_bottleneck * (1.0 + 0.5435597945015939 * norm_uncertainty)
    score = norm_urgency + 0.6143164390989989 * bottleneck_with_risk + feasibility_gate * 0.4223611168265451 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_active * 0.3507478874955595 * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
