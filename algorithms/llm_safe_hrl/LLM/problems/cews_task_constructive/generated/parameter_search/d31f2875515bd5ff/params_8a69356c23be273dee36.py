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
    eps = 1.0432694319351788e-06
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
        q_low = np.percentile(x, 15.903961218715423)
        q_high = np.percentile(x, 83.06646800446599)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    slack_urgency = 1.0 / (1.0 + np.exp(-1.9408689565660955 * (slack - 0.7985319493321259)))
    slack_norm = iqr_normalize(slack)
    slack_pressure = np.maximum(-slack_norm, 0.0)
    successor_delay = np.clip(remaining_work / (slack + eps), 0.0, 12.06143169767176)
    norm_successor_delay = iqr_normalize(successor_delay)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    ddl_protection_gate = np.where((rank_percentile >= 0.8929757312107729) & (slack < np.median(slack)), 1.0, 0.0)
    coupled_urgency = 0.43733744105490124 * slack_pressure * ddl_protection_gate
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    wait_scaled = ready_wait_time / (10.096929718991214 + eps)
    norm_wait = 2.0 / np.pi * np.arctan(wait_scaled)
    score = slack_urgency * 2.0 + 2.274849907718035 * norm_successor_delay + coupled_urgency + 0.10855310150119876 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
