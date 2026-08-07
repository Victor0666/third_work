import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's smooth urgency and bottleneck coupling
    with Parent 1's robust DDL-protection semantics and enhanced uncertainty-slack interaction.
    
    Key structural improvements:
      - Replaces binary DDL-protection gate with graded uncertainty-slack interaction term,
        enabling proportional amplification of urgency under risk.
      - Introduces explicit multiplicative coupling: norm_uncertainty * norm_urgency, weighted by tunable parameter.
      - Tightens IQR percentiles (20/80) for more responsive normalization.
      - Retains arctan wait saturation and sigmoid urgency for stability and smoothness.
      - Keeps successor bottleneck coupling and critical-path gating for path-aware scheduling.
      - All terms remain finite, deterministic, and shape-preserving.
    """
    eps = 0.08478607212582288
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
        q_low = np.percentile(x, 11.531813850404317)
        q_high = np.percentile(x, 71.30307466724727)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    slack_centered = slack - 0.038133239285233156
    sigmoid_input = np.clip(-6.643581948153861 * slack_centered, -38.035247774721256, 38.035247774721256)
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
    critical_gate = np.where(rank_percentile >= 0.7470788528772612, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (5.795199237243685 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = iqr_normalize(wait_saturation)
    norm_uncertainty = iqr_normalize(uncertainty)
    uncertainty_slack_interaction = 0.3054318338836154 * norm_uncertainty * norm_urgency
    score = norm_urgency + 0.6077692381715467 * norm_bottleneck + 0.8868292442161739 * norm_energy_eff - critical_bonus - norm_wait + uncertainty_slack_interaction
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
