import numpy as np

def get_task_priority_v2(
    min_exec_time,
    min_comm_time,
    min_incremental_energy,
    slack,
    upward_rank,
    remaining_work,
    ready_wait_time,
    uncertainty
):
    """
    Self-evolved priority rule: fixes global scaling distortion and unfair damping,
    unifies uncertainty treatment, strengthens deadline compliance *per-task*,
    and refines energy-efficiency trade-off with tighter normalization.
    
    Key improvements:
      - Replaces global `has_late_task` scaling with per-task arctan penalty boosted by |slack| when negative,
      - Restores fairness for late tasks: wait boost is *not* damped by uncertainty (urgent waiting matters most),
      - Applies uncertainty inflation to latency *unconditionally*, but bounded & normalized to avoid over-penalization,
      - Introduces slack-aware critical-energy gating: higher weight when slack is tight (0 < slack < 5s) vs. ample,
      - Uses variance-stabilized normalization for energy ratio to improve robustness on sparse/low-energy tasks,
      - Adds minimal but decisive penalty for zero-slack tasks (hard deadline boundary).
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    def safe_mad_normalize(x):
        """Robust MAD normalization: handles N=1, constants, outliers; clips to [-4,4]"""
        if x.size == 1:
            return np.zeros_like(x)
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev) + eps
        normed = (x - med) / mad
        return np.clip(normed, -4.0, 4.0)
    
    # Per-task deadline risk: arctan(-slack) for late tasks, enhanced linear penalty near zero slack
    deadline_risk_raw = np.where(slack < 0, np.arctan(-slack), 0.0)
    # Boost penalty for tasks at or near hard deadline boundary (slack <= 0.1s)
    zero_slack_penalty = np.where(slack <= 0.1, 1.5, 0.0)
    deadline_score = safe_mad_normalize(deadline_risk_raw + zero_slack_penalty)
    
    # Critical work only active when slack > 0; normalized
    critical_work_gated = np.where(slack > 0, upward_rank * remaining_work, 0.0)
    critical_work_norm = safe_mad_normalize(critical_work_gated)
    
    # Critical-energy efficiency: gated by slack > 0, with tighter weighting for tight slack (0 < slack < 5)
    energy_denom = min_incremental_energy + eps
    base_ratio = (upward_rank * remaining_work + eps) / energy_denom
    # Apply stronger weight when slack is tight (0 < slack < 5s) → higher priority for energy-efficient critical tasks
    slack_weight = np.where((slack > 0) & (slack < 5), 1.2, 1.0)
    critical_energy_ratio = np.where(slack > 0, base_ratio * slack_weight, 0.0)
    # Variance-stabilized sigmoid: avoids saturation for extreme ratios
    clipped_ratio = np.clip(critical_energy_ratio, -6.0, 6.0)
    critical_energy_sigmoid = 1.0 / (1.0 + np.exp(-clipped_ratio * 0.25))
    critical_energy_norm = safe_mad_normalize(critical_energy_sigmoid)
    
    # Waiting fairness: sqrt-based, scaled by slack margin — NOT damped by uncertainty (urgent waiting must win)
    wait_boost_base = np.sqrt(np.clip(ready_wait_time, 0.0, None))
    slack_margin = np.clip(slack, 0.0, None) + eps
    wait_boost = np.clip(wait_boost_base / (slack_margin + wait_boost_base + eps), 0.0, 0.25)
    
    # Uncertainty inflates *all* latency uniformly (no conditional logic), but bounded & normalized
    total_latency = min_exec_time + min_comm_time + eps
    uncertainty_inflated_latency = total_latency * (1.0 + np.clip(uncertainty, 0.0, 3.0) * 0.15)
    inflated_lat_norm = safe_mad_normalize(uncertainty_inflated_latency)
    
    # Remaining work penalty (small, consistent)
    remaining_work_norm = safe_mad_normalize(remaining_work) * 0.02
    
    # Final score: deadline dominates (4.0), critical work secondary (-2.0), energy efficiency positive (+0.85),
    # wait boost positive (+0.2), latency inflation positive (+0.2), work penalty tiny (+0.02)
    score = (
        +4.0 * deadline_score
        - 2.0 * critical_work_norm
        + 0.85 * critical_energy_norm
        + 0.20 * wait_boost
        + 0.20 * inflated_lat_norm
        + remaining_work_norm
    )
    
    return np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
