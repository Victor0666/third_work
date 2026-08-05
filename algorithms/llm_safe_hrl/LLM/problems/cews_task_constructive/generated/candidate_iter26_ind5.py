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
    v4 priority rule: Hybrid urgency-criticality-energy optimizer with degenerate-safe normalization,
                      adaptive slack gating, starvation rescue under positive slack only,
                      and risk-normalized energy density.

    Key innovations:
    - Combines Parent 2's robust median-based adaptive slack threshold and degenerate-safe minmax.
    - Adopts Parent 1's risk-adjusted duration (duration * (1 + uncertainty)) for energy density,
      preserving fidelity under volatility.
    - Integrates criticality-gated energy penalty via sigmoid on (upward_rank * rel_slack) for smooth
      importance modulation near deadlines.
    - Starvation rescue uses wait_ratio > 1.5*median_ratio AND slack > 0 AND upward_rank > median_rank,
      preventing late-task bias while prioritizing starved critical small tasks.
    - Uncertainty boost is gated by both proximity_bias (exp(-max(0,slack)/tau)) AND dur_uncertainty,
      capped at 1.0, and normalized robustly.
    - Energy penalty mask requires tight slack, high rank, AND significant energy density (> p10),
      ensuring only meaningful energy contributors are penalized.
    - Final weights: urgency (0.45), critical latency (0.20), gated energy (0.15), fairness (0.08),
                     starvation (0.06), uncertainty (0.03), residual work (0.03).
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_finite = x[np.isfinite(x)]
        if x_finite.size == 0:
            return np.zeros_like(x)
        if x_finite.size < 3:
            x_min, x_max = np.min(x_finite), np.max(x_finite)
        else:
            p01 = np.percentile(x_finite, 1.0, method='midpoint')
            p99 = np.percentile(x_finite, 99.0, method='midpoint')
            x_clipped = np.clip(x_finite, p01, p99)
            x_min, x_max = np.min(x_clipped), np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        normed = np.zeros_like(x)
        valid_mask = np.isfinite(x)
        normed[valid_mask] = (x[valid_mask] - x_min) / (x_max - x_min + eps)
        return normed

    # Compute duration and risk-adjusted duration
    duration = min_exec_time + min_comm_time + eps
    risk_adjusted_duration = duration * (1.0 + uncertainty)
    
    # Energy density: risk-normalized (energy per risk-weighted duration)
    energy_density = np.divide(min_incremental_energy, risk_adjusted_duration, 
                               out=np.zeros_like(min_incremental_energy), 
                               where=risk_adjusted_duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Slack-based urgency and gating signals
    is_urgent = (slack <= 0.0).astype(float)
    positive_slack = slack[slack > 0]
    adaptive_slack_thresh = np.median(positive_slack) if len(positive_slack) > 0 else 1.0
    tight_slack_mask = (slack <= adaptive_slack_thresh).astype(float)
    
    # Critical latency: duration scaled by upward_rank coupling
    norm_upward_rank = robust_minmax_norm(upward_rank)
    critical_latency_raw = duration * (1.0 + 0.5 * norm_upward_rank)
    norm_critical_latency = robust_minmax_norm(critical_latency_raw)
    
    # Gated energy penalty: only when slack is tight, rank is high, and energy density is significant
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_density_finite = energy_density[np.isfinite(energy_density)]
    energy_floor = np.percentile(energy_density_finite, 10) + eps if len(energy_density_finite) > 0 else eps
    energy_significant_mask = (energy_density >= energy_floor).astype(float)
    high_rank_mask = (upward_rank >= np.median(upward_rank) + eps).astype(float) if N > 1 else np.ones(N)
    energy_penalty_mask = tight_slack_mask * high_rank_mask * energy_significant_mask
    
    # Criticality-gated energy: sigmoid on (upward_rank * clipped relative slack)
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    rel_slack_clipped = np.clip(rel_slack, 0.0, 1.0)
    criticality_gate_input = upward_rank * rel_slack_clipped
    criticality_gate = 1.0 / (1.0 + np.exp(-(criticality_gate_input - 0.5)))
    gated_energy_penalty = norm_energy_density * energy_penalty_mask * criticality_gate
    
    # Fairness term: wait_per_work normalized, activated only for non-urgent tasks
    wait_per_work = np.divide(ready_wait_time, remaining_work + eps, 
                              out=np.zeros_like(ready_wait_time), 
                              where=remaining_work + eps != 0)
    wait_per_work = np.nan_to_num(wait_per_work, nan=0.0, posinf=0.0, neginf=0.0)
    norm_wait_per_work = robust_minmax_norm(wait_per_work)
    wpw_finite = wait_per_work[np.isfinite(wait_per_work)]
    wait_threshold = np.median(wpw_finite) * 1.5 + eps if len(wpw_finite) > 0 else eps
    wait_gate = (wait_per_work >= wait_threshold).astype(float)
    wait_penalty = (1.0 - is_urgent) * norm_wait_per_work * wait_gate
    
    # Starvation rescue: only when slack > 0, wait_ratio > 1.5*median_ratio, and rank is high
    wait_ratio = np.divide(ready_wait_time, duration + eps, 
                           out=np.zeros_like(ready_wait_time), 
                           where=duration + eps != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    ratio_finite = wait_ratio[np.isfinite(wait_ratio)]
    median_ratio = np.median(ratio_finite) if len(ratio_finite) > 0 else 1.0
    is_starvable = (wait_ratio > 1.5 * median_ratio) & (slack > 0.0) & (upward_rank >= np.median(upward_rank))
    starvation_boost = np.where(is_starvable, wait_ratio * (1.0 + 0.1 * norm_upward_rank), 0.0)
    norm_starvation = robust_minmax_norm(starvation_boost)
    
    # Uncertainty boost: gated by proximity and duration uncertainty, capped and normalized
    tau = 15.0
    proximity_bias = np.exp(-np.maximum(0.0, slack) / tau)
    dur_uncertainty = np.divide(uncertainty, duration, 
                                out=np.zeros_like(uncertainty), 
                                where=duration != 0)
    dur_uncertainty = np.nan_to_num(dur_uncertainty, nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty_boost = np.clip(dur_uncertainty * proximity_bias, 0.0, 1.0)
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Residual work term: normalized remaining_work, inversely weighted (smaller work → higher priority when fair)
    norm_remaining_work = robust_minmax_norm(remaining_work)
    
    # Base score composition
    base_score = (
        0.20 * norm_critical_latency +
        0.15 * gated_energy_penalty +
        0.08 * wait_penalty +
        0.06 * norm_starvation +
        0.03 * norm_uncertainty_boost +
        0.03 * norm_remaining_work
    )
    
    # Urgency dominates: assign massive negative priority to urgent tasks
    score = np.where(is_urgent, -1000000000000.0, base_score + 0.45)
    
    # Clamp and sanitize
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
