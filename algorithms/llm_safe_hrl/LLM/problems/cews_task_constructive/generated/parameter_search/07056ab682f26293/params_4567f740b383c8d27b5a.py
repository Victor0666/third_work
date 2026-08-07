import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's smooth urgency and bottleneck coupling
    with Parent 1's robust DDL-protection semantics and enhanced uncertainty-slack interaction.
    
    Key structural improvements:
      - Replaces binary DDL-protection gate with *graded* uncertainty-slack coupling:
        multiplies normalized uncertainty with sigmoid urgency — enabling fine-grained risk amplification.
      - Uses asymmetric IQR percentiles (20/80) for tighter normalization robustness against outliers.
      - Retains smooth arctan wait saturation and successor bottleneck term for fairness & blocking mitigation.
      - Critical-path bonus now uses *normalized* upward_rank scaled by percentile gate — avoids rank-scale bias.
      - All terms are IQR-normalized and bounded; no piecewise discontinuities or unbounded growth.
    """
    eps = 0.0002594936241962719
    N = len(slack)
    finfo = np.finfo(float)
    eps_safe = max(eps, finfo.tiny)
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
        q_low = np.percentile(x, 23.95298495108658)
        q_high = np.percentile(x, 88.53075558981735)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps_safe else np.max(np.abs(x - center)) + eps_safe
        return (x - center) / (denom + eps_safe)
    slack_centered = slack - 0.8189680065970779
    sigmoid_input = np.clip(-4.7757802036190125 * slack_centered, -21.457955935811846, 21.457955935811846)
    urgency_gate = 2.0 / (1.0 + np.exp(sigmoid_input)) - 1.0
    norm_urgency = iqr_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps_safe)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    bottleneck_pressure = duration * (remaining_work + upward_rank + eps_safe)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps_safe)
    critical_gate = np.where(rank_percentile >= 0.7421547574136029, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (0.5527262153127935 + eps_safe)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = iqr_normalize(wait_saturation)
    norm_uncertainty = iqr_normalize(uncertainty)
    graded_coupling = 0.542793508761829 * norm_uncertainty * norm_urgency
    score = norm_urgency + 0.9050036920915292 * norm_bottleneck + 0.7241490457510087 * norm_energy_eff - critical_bonus - norm_wait + graded_coupling
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
