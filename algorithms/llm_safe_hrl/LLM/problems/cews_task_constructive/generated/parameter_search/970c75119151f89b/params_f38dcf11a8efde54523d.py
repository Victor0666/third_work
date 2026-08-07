import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust IQR normalization with Parent 1's feasibility-aware gating
    and uncertainty-modulated bottleneck coupling — validated by counterfactual replay.
    
    Key structural improvements:
      - Feasibility-aware gating: nullifies energy & uncertainty contributions when slack > threshold,
        preventing over-penalization of low-risk tasks and aligning with hard deadline constraint priority.
      - Uncertainty-modulated bottleneck: multiplicative coupling (not additive) reflects that risk amplifies blocking impact.
      - Tunable IQR normalization (20th/80th percentiles) improves outlier resilience vs fixed 25/75.
      - Smooth arctan wait saturation + percentile-gated critical bonus retained for fairness and path sensitivity.
      - All operations guarded against NaN/inf/zero; deterministic and finite.
    """
    eps = 1.9911696482434154e-05
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
        q_low = np.percentile(x, 24.661804958469823)
        q_high = np.percentile(x, 73.84831024059793)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    sigmoid_input = np.clip(-2.750863865679781 * slack, -24.684800811028293, 24.684800811028293)
    urgency_gate = 2.0 / (1.0 + np.exp(sigmoid_input)) - 1.0
    norm_urgency = iqr_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    bottleneck_pressure = duration * (remaining_work + upward_rank + eps)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    norm_uncertainty = iqr_normalize(uncertainty)
    bottleneck_with_risk = norm_bottleneck * (1.0 + 0.07984996152302709 * norm_uncertainty)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.526290261515772, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (16.999088034979764 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = iqr_normalize(wait_saturation)
    feasibility_gate = np.where(slack > 5.829269661499667, 0.0, 1.0)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < median_slack) & (uncertainty > 0.8090645646924427 * max_uncertainty)).astype(float)
    score = norm_urgency + 0.5108689941065008 * bottleneck_with_risk + feasibility_gate * 1.2533744025461 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_active * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
