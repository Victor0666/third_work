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
    Hybrid priority rule v2: Combines v1's robust slack-gated urgency & starvation fairness
    with v2's percentile scaling, adaptive criticality-energy tradeoff, and work-aware modulation.
    
    Key innovations:
    - Uses v2's monotonic percentile_scale for all features (improves consistency & reduces variance)
    - Retains v1's slack_tightness_gate to amplify adjustments only when deadlines are tight
    - Integrates v2's beta-gated (1+uncertainty)^beta criticality amplification under violation
    - Replaces v1's IQR normalization and v2's raw percentile scaling with *bounded percentile rank*
      using argsort(argsort) + epsilon-clipped 0–1 mapping (more stable for small N)
    - Introduces *energy-density slack sensitivity*: scales energy_density contribution by
      exp(-max(0, slack)/median_duration) to suppress energy optimization when slack is positive
    - Adds *uncertainty-aware communication penalty*: min_comm_time * uncertainty * (abs_slack < median_abs_slack)
      only for tasks where communication dominates execution (min_comm_time > 0.8 * duration)
    - All operations eps-protected; nan/inf guarded; deterministic; shape-(N,) guaranteed.
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
    
    # Precompute common terms
    duration = min_exec_time + min_comm_time + eps
    median_duration = np.median(duration) + eps
    median_abs_slack = np.median(np.abs(slack)) + eps
    median_rw = np.median(remaining_work) + eps
    
    # Bounded percentile scaling: monotonic, [0,1]-bounded, robust for small N
    def percentile_scale(x):
        if len(x) == 1:
            return np.array([0.5])
        ranks = np.argsort(np.argsort(x)) + 1.0
        return np.clip(ranks / (len(x) + 1.0), 0.0, 1.0)
    
    # Slack-tightness gate: amplifies priority adjustments only near deadline
    abs_slack = np.abs(slack)
    slack_tightness_gate = 1.0 / (1.0 + np.exp(-2.0 * (median_abs_slack - abs_slack) / median_abs_slack))
    
    # Deadline urgency: normalized slack deficit modulated by work impact and tightness
    slack_deficit = np.maximum(0.0, -slack)
    normalized_slack_deficit = np.clip(slack_deficit / median_duration, 0.0, 10.0)
    work_impact_factor = np.where(slack <= 0.0, np.clip(remaining_work / median_rw, 0.3, 3.0), 1.0)
    deadline_urgency = normalized_slack_deficit * work_impact_factor * 0.75 + \
                       (1.0 - np.exp(-np.clip(slack_deficit, 0.0, 5.0))) * 0.25
    urgency_norm = percentile_scale(deadline_urgency)
    
    # Energy density: min_incremental_energy per effective duration
    energy_density = np.clip(min_incremental_energy / duration, eps, 1e6)
    # Slack-sensitive energy contribution: suppressed when slack > 0
    energy_slack_weight = np.exp(-np.clip(slack, 0.0, 100.0) / median_duration)
    energy_density_weighted = energy_density * energy_slack_weight
    energy_norm = percentile_scale(energy_density_weighted)
    
    # Criticality-energy tradeoff: upward_rank * (1+uncertainty)^beta / energy_density
    beta = np.where(slack <= 0.0, 1.5, 0.5)
    risk_factor = np.power(1.0 + np.clip(uncertainty, 0.0, 10.0), beta)
    crit_eff_score = upward_rank * risk_factor / (energy_density + eps)
    crit_eff_norm = percentile_scale(np.clip(crit_eff_score, eps, 1e7))
    
    # Starvation relief: log-scaled wait time, gated by slack sign and criticality
    wait_log = np.log1p(ready_wait_time)
    wait_penalty = np.where(slack <= 0.0, 0.0,
                           np.clip(percentile_scale(wait_log) * (1.0 - percentile_scale(upward_rank)), 0.0, 0.2))
    
    # Uncertainty-aware communication penalty: only when comm dominates and slack is tight
    comm_dominance = (min_comm_time > 0.8 * duration).astype(float)
    comm_penalty = min_comm_time * uncertainty * comm_dominance * (abs_slack < median_abs_slack).astype(float)
    comm_norm = percentile_scale(comm_penalty)
    
    # Duration and uncertainty baselines
    dur_norm = percentile_scale(duration)
    unc_norm = percentile_scale(uncertainty)
    
    # Final weighted score: smaller = higher priority
    # Weights tuned to emphasize urgency (0.4), critical-efficiency (0.3), energy (0.15), 
    # starvation (0.05), comm penalty (0.05), duration (0.03), uncertainty (0.02)
    score = (
        0.4 * urgency_norm +
        0.3 * (1.0 - crit_eff_norm) +  # invert: higher crit_eff => lower priority score
        0.15 * energy_norm +
        0.05 * wait_penalty +
        0.05 * comm_norm +
        0.03 * dur_norm +
        0.02 * unc_norm
    )
    
    # Apply slack-tightness gating to sharpen focus near deadlines
    score = score * slack_tightness_gate
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    
    return score
