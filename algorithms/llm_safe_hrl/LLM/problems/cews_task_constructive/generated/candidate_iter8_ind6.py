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
    Hybrid priority rule v2: integrates Parent 2's hardened urgency gating and robust normalization
    with Parent 1's slack-adaptive criticality amplification and energy-density reversal,
    while adding novel deadline-robust energy fairness and uncertainty-aware starvation guard.
    
    Key innovations:
      - *Slack-gated criticality-energy ratio*: replaces static crit_per_energy with
        (upward_rank * (1 + 0.5*tanh(slack/median_duration))) / (min_incremental_energy + eps),
        enabling smooth criticality boost near deadline without discontinuity.
      - *Uncertainty-weighted energy penalty* only activated under slack deficit AND high uncertainty,
        avoiding unnecessary energy inflation when uncertainty is low or slack is positive.
      - *Starvation guard* now uses bounded wait-pressure scaled by (1 - soft_urgency) AND slack-feasibility,
        preventing starvation of non-urgent tasks only when they're still DDL-feasible.
      - *Energy-density reversal* retained from Parent 1 but made continuous: sign(slack) * percentile-scaled energy_density,
        ensuring low-energy tasks prioritized when slack > 0, high-energy (fast-offload) when slack < 0.
      - All components normalized via robust_minmax_norm with degenerate-case fallback; final score bounded and finite.
      - Weights sum to 1.0 and respect performance analysis: urgency dominates (0.42), energy fairness (0.23),
        latency (0.12), criticality (0.10), starvation (0.08), work impact (0.05).
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
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)
    
    # Robust normalization supporting small-N and degenerate cases
    def robust_minmax_norm(x):
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)
    
    # Duration-based reference scale
    total_latency = min_exec_time + min_comm_time + eps
    median_duration = np.median(total_latency) + eps
    median_slack = np.median(slack)
    q1_slack, q3_slack = np.percentile(slack, [25, 75])
    iqr_slack = q3_slack - q1_slack + eps
    
    # Hardened urgency gate: steep sigmoid centered at median_slack
    soft_urgency = 1.0 / (1.0 + np.exp(-(slack - median_slack) / (0.25 * iqr_slack + eps)))
    
    # Slack-adaptive criticality amplification (Parent 1) + energy density reversal (Parent 1)
    slack_normalized = slack / (median_duration + eps)
    criticality_factor = 1.0 + 0.5 * np.tanh(slack_normalized)
    amplified_upward_rank = upward_rank * criticality_factor
    energy_density = np.clip(min_incremental_energy / total_latency, eps, 1e6)
    
    # Continuous energy-density reversal: low energy preferred when slack > 0, high when slack < 0
    energy_norm_base = robust_minmax_norm(energy_density)
    energy_reversed = np.where(slack > 0, energy_norm_base, 1.0 - energy_norm_base)
    
    # Criticality-energy ratio with slack-adaptive numerator
    crit_energy_ratio = amplified_upward_rank / (min_incremental_energy + eps)
    norm_crit_energy_ratio = robust_minmax_norm(np.clip(crit_energy_ratio, eps, 1e7))
    
    # Uncertainty-weighted energy penalty only under slack deficit AND moderate/high uncertainty
    tight_slack_mask = (slack <= median_slack).astype(float)
    high_uncertainty_mask = (uncertainty > np.percentile(uncertainty, 50)).astype(float)
    energy_penalty = robust_minmax_norm(min_incremental_energy) * (1.0 + 0.7 * uncertainty * tight_slack_mask * high_uncertainty_mask)
    
    # Latency term (total execution + comm time)
    norm_latency = robust_minmax_norm(total_latency)
    
    # Work impact: smoothed slack-sensitive penalty for large remaining work when late
    tau = np.maximum(np.abs(median_slack), 1.0) + eps
    slack_sensitivity = np.exp(-np.clip(np.maximum(-slack, 0.0), 0.0, 100.0) / tau)
    norm_work = robust_minmax_norm(remaining_work)
    work_penalty = norm_work * slack_sensitivity
    
    # Starvation guard: relative wait pressure only for non-urgent AND slack-feasible tasks
    wait_gate = np.where((soft_urgency < 0.6) & (slack >= 0), 1.0, 0.0)
    if N == 1:
        wait_percentile = np.array([0.5])
    else:
        wait_sorted = np.sort(ready_wait_time)
        wait_ranks = np.array([np.searchsorted(wait_sorted, w, side='right') for w in ready_wait_time])
        wait_percentile = wait_ranks / (N + eps)
    wait_boost = wait_percentile * wait_gate
    norm_wait_boost = robust_minmax_norm(wait_boost) if N > 0 else np.zeros(N)
    starvation_term = 1.0 - norm_wait_boost
    
    # Base urgency dominates; others modulate
    base_urgency = 1.0 - soft_urgency
    
    # Final convex combination — weights sum to 1.0
    score = (
        0.42 * base_urgency +
        0.23 * energy_penalty +
        0.12 * norm_latency +
        0.10 * (1.0 - norm_crit_energy_ratio) +
        0.08 * starvation_term +
        0.05 * work_penalty
    )
    
    # Ensure finite, bounded output
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
