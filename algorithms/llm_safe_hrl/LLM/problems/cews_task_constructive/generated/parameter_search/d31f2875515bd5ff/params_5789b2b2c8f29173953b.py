import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's robust structure with Parent 1's stability,
       enhanced by smooth sigmoid urgency, successor-aware delay penalty, and conditional DDL-protection.
    
    Key innovations:
      - Smooth, differentiable sigmoid urgency gate on slack (replaces piecewise), centered at configurable offset
      - Successor release delay term: `min(remaining_work / (slack + eps), successor_delay_upper_bound)` explicitly penalizes bottleneck tasks
      - DDL-protection gate: activates only when task is both high-rank *and* slack is below median (tight)
      - Critical-path coupling: multiplies normalized slack pressure with rank to amplify urgency on critical path
      - Arctan-saturated wait boost with tunable scale for fairness without starvation
      - All features normalized via configurable IQR percentiles for robustness to outliers
    """
    eps = 0.0020430599069364078
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
        q_low = np.percentile(x, 15.705526533086148)
        q_high = np.percentile(x, 78.69930913286264)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    slack_urgency = 1.0 / (1.0 + np.exp(-2.4802961301926505 * (slack - 0.6797939501074541)))
    slack_norm = iqr_normalize(slack)
    slack_pressure = np.maximum(-slack_norm, 0.0)
    successor_delay = np.clip(remaining_work / (slack + eps), 0.0, 11.87276096663391)
    norm_successor_delay = iqr_normalize(successor_delay)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    ddl_protection_gate = np.where((rank_percentile >= 0.7899566544755202) & (slack < np.median(slack)), 1.0, 0.0)
    coupled_urgency = 0.4640381287662993 * slack_pressure * ddl_protection_gate
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    wait_scaled = ready_wait_time / (6.649977734925434 + eps)
    norm_wait = 2.0 / np.pi * np.arctan(wait_scaled)
    score = slack_urgency * 2.0 + 2.2872470872375863 * norm_successor_delay + coupled_urgency + 0.6946403636865708 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
