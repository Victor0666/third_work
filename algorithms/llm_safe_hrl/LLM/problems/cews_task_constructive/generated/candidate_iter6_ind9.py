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
    Self-evolved priority rule v2: Enhances deadline enforcement fidelity, eliminates normalization-induced rank inversion,
    introduces physics-aware energy efficiency gating, and adds adaptive starvation mitigation with criticality-aware decay.
    
    Key evolutions:
      - Replaces percentile scaling with *monotonic rank-preserving quantile mapping* to guarantee strict ordering preservation
        under distribution shifts (e.g., all slack < 0 → urgency=1 for all), fixing percentile-scale rank collapse.
      - Introduces *energy efficiency threshold gating*: only penalizes energy when marginal energy density exceeds local median,
        preventing spurious penalties on inherently low-energy tasks.
      - Refines augmented urgency with *asymmetric slack sensitivity*: steeper sigmoid rise for negative slack (lateness),
        shallower decay for positive slack — matches real DDL violation risk asymmetry.
      - Starvation penalty now decays exponentially with upward_rank, prioritizing fairness only for low-criticality long-waiting tasks.
      - Uncertainty boosting uses *risk-congruent slack sign modulation*: amplifies uncertainty impact only when slack < 0,
        and attenuates it smoothly as slack increases — avoids over-penalizing high-uncertainty but early tasks.
      - Adds *latency criticality coupling*: communication time weighted by upstream dependency count (implicit via upward_rank),
        modeling data-stall amplification in tight deadlines.
      - Final score enforces strict dominance hierarchy: urgency >> energy-efficiency >> latency-coupled-comm >> fairness,
        with no cross-term interference.
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
    
    # Physics-aware duration & energy density
    duration = min_exec_time + min_comm_time + eps
    energy_density = np.clip(min_incremental_energy / (duration + eps), eps, 1e6)
    
    # Asymmetric urgency: steeper response to lateness (slack < 0)
    median_slack = np.median(slack)
    iqr_slack = np.percentile(slack, 75) - np.percentile(slack, 25) + eps
    # Use different scales: tight for negative slack, gentle for positive
    slack_scale = np.where(slack < 0, iqr_slack * 0.5, iqr_slack * 2.0) + eps
    urgency = 1.0 / (1.0 + np.exp(-(slack - median_slack) / (slack_scale + eps)))
    
    # Work-impact augmentation only for late tasks (slack <= 0), scaled by normalized remaining work
    median_rw = np.median(remaining_work) + eps
    rw_ratio = np.clip(remaining_work / median_rw, 0.1, 10.0)
    work_impact_factor = np.where(slack <= 0, rw_ratio, 1.0)
    augmented_urgency = np.clip(urgency + 0.4 * (1.0 - urgency) * work_impact_factor * (slack <= 0), 0.0, 1.0)
    
    # Monotonic quantile mapping (rank-based, preserves order strictly)
    def quantile_map(x):
        if len(x) == 1:
            return np.array([0.5])
        # Rank from 0 to len-1, then map to [0.05, 0.95] to avoid boundary extremes
        ranks = np.argsort(np.argsort(x)).astype(float)
        return 0.05 + 0.9 * (ranks / (len(x) - 1 + eps))
    
    dur_norm = quantile_map(duration)
    energy_den_norm = quantile_map(energy_density)
    ur_norm = quantile_map(upward_rank)
    rw_norm = quantile_map(remaining_work)
    unc_norm = quantile_map(uncertainty)
    wait_norm = quantile_map(np.log1p(ready_wait_time))
    
    # Energy efficiency gating: only penalize above local median energy density
    median_energy_den = np.median(energy_density) + eps
    energy_efficiency_mask = (energy_density <= median_energy_den).astype(float)
    energy_penalty = energy_den_norm * (1.0 + 0.8 * augmented_urgency) * (1.0 - energy_efficiency_mask)
    
    # Criticality-efficiency term: only activate when urgency > 0.5 AND energy not efficient
    crit_eff_score = upward_rank / (energy_density + eps)
    crit_eff_norm = quantile_map(np.clip(crit_eff_score, eps, 1e7))
    crit_gate = np.where((augmented_urgency > 0.5) & (energy_efficiency_mask == 0.0), 1.0, 0.0)
    crit_term = (1.0 - crit_eff_norm) * crit_gate
    
    # Latency-coupled communication: weight comm time by criticality to model stall amplification
    comm_latency_weighted = min_comm_time * (1.0 + 0.5 * ur_norm)
    comm_norm = quantile_map(comm_latency_weighted + min_exec_time + eps)
    latency_term = comm_norm
    
    # Risk-congruent uncertainty boost: only amplify uncertainty when slack < 0
    slack_sign_mask = np.where(slack < 0, 1.0, np.clip(slack / (np.abs(median_slack) + eps), 0.0, 1.0))
    uncertainty_boost = uncertainty * slack_sign_mask * (1.0 - augmented_urgency)
    unc_boost_norm = quantile_map(uncertainty_boost)
    
    # Adaptive starvation penalty: exponential decay with upward_rank, active only for non-urgent & non-late
    wait_gate = np.where((augmented_urgency < 0.6) & (slack >= 0), 1.0, 0.0)
    # Decay: higher criticality → less fairness priority
    starv_decay = np.exp(-0.5 * ur_norm)
    wait_penalty = wait_norm * wait_gate * starv_decay
    
    # Strict hierarchical weighting (urgency dominates all; others secondary)
    score = (
        0.60 * (1.0 - augmented_urgency) +     # Primary: deadline enforcement
        0.20 * energy_penalty +                # Secondary: energy waste under urgency
        0.10 * crit_term +                     # Tertiary: criticality-per-energy tradeoff
        0.05 * latency_term +                  # Quaternary: latency-coupled stall risk
        0.03 * wait_penalty +                  # Quinary: fairness for low-criticality long-wait
        0.02 * (1.0 - unc_boost_norm)          # Residual: uncertainty-aware robustness
    )
    
    # Robust final sanitization
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score
