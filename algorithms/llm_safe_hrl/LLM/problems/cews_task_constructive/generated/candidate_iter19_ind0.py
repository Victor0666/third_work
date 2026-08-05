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
    v2 priority rule: Hard urgency dominance + normalized critical-path latency + risk-gated energy penalty +
                      adaptive starvation relief + uncertainty-weighted slack compression.
    
    Key mutations from v1:
    - Replaces tanh-scaled slack with robust clipped inverse slack (no asymptotes, bounded sensitivity)
    - Introduces *risk-aware slack compression*: uncertainty modulates slack's effective tightness via multiplicative gating
    - Uses trimmed-mean + MAD normalization instead of percentile clipping for better small-N stability
    - Energy penalty now gated by both upward_rank *and* normalized slack (not binary rel_slack threshold)
    - Starvation relief uses wait-time *relative to median task duration*, not just raw wait time
    - Adds light "work density" term: energy per MI, normalized and weighted only for non-urgent tasks
    - All components scaled to [0,1] range before combination; no unbounded coefficients
    - Explicit NaN/inf guard at every intermediate step, with deterministic fallbacks
    """
    eps = 1e-8
    # Ensure float64 & copy to avoid mutation
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)
    
    # Robust MAD-based normalization (more stable than percentile for N<10)
    def robust_mad_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        med = np.median(x_clean)
        mad = np.median(np.abs(x_clean - med))
        if mad < eps:
            return np.zeros_like(x_clean)
        normed = (x_clean - med) / (mad + eps)
        # Clip to [-3, 3] → then squash to [0,1] via sigmoid-like tanh+shift
        clipped = np.clip(normed, -3.0, 3.0)
        return 0.5 * (np.tanh(clipped / 2.0) + 1.0)  # maps [-3,3] → ~[0.03,0.97]
    
    # Urgency flag: hard constraint — zero-tolerance for slack <= 0
    is_urgent = (slack <= 0.0).astype(np.float64)
    
    # Duration base (exec + comm), avoid zero
    duration = min_exec_time + min_comm_time + eps
    
    # Risk-compressed slack: uncertainty scales slack's urgency effect multiplicatively
    # Higher uncertainty → tighter effective deadline (smaller slack magnitude counts more)
    compressed_slack = slack * (1.0 + 0.5 * uncertainty)  # boosts negative slack impact
    compressed_slack = np.nan_to_num(compressed_slack, nan=0.0, posinf=0.0, neginf=-1e12)
    
    # Inverse slack sensitivity: bounded reciprocal (no division by zero or inf)
    abs_compressed_slack = np.abs(compressed_slack) + eps
    slack_sensitivity = np.clip(1.0 / abs_compressed_slack, 0.01, 100.0)
    # Normalize sensitivity to [0,1] — high sensitivity = very tight slack
    norm_slack_sensitivity = robust_mad_norm(slack_sensitivity)
    
    # Critical latency: duration weighted by upward rank, normalized
    critical_latency = duration * (1.0 + 0.8 * upward_rank)
    norm_critical_latency = robust_mad_norm(critical_latency)
    
    # Energy density: incremental energy per unit duration
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_mad_norm(energy_density)
    
    # Risk-gated energy penalty: only activates when both slack is tight AND rank is high
    # Use continuous gating: sigmoid of normalized slack + rank product
    gate_input = norm_slack_sensitivity * robust_mad_norm(upward_rank)
    energy_gate = 1.0 / (1.0 + np.exp(-4.0 * (gate_input - 0.5)))  # smooth [0,1] gate
    energy_penalty = norm_energy_density * energy_gate
    
    # Starvation relief: wait time relative to *median task duration*, not absolute
    median_duration = np.median(duration) + eps
    wait_ratio = np.divide(ready_wait_time, median_duration, out=np.zeros_like(ready_wait_time), where=median_duration != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    norm_wait_ratio = robust_mad_norm(wait_ratio)
    # Only apply starvation relief to non-urgent tasks (urgent already prioritized)
    wait_penalty = (1.0 - is_urgent) * norm_wait_ratio
    
    # Work density term: energy per MI (for non-urgent, high-work tasks)
    work_density = np.divide(min_incremental_energy, remaining_work + eps, out=np.zeros_like(min_incremental_energy), where=remaining_work + eps != 0)
    work_density = np.nan_to_num(work_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_work_density = robust_mad_norm(work_density)
    work_density_penalty = (1.0 - is_urgent) * norm_work_density * robust_mad_norm(remaining_work)
    
    # Base score: all components start at 0; urgent tasks get fixed minimum
    score = np.full(N, 1.0, dtype=np.float64)
    score = np.where(is_urgent, -1e12, score)
    
    # Combine non-urgent contributions with balanced weights (sum to 1.0)
    # Weight order reflects hierarchy: urgency > critical latency > energy > fairness > work density
    score = np.where(
        is_urgent,
        score,
        score + 
        0.35 * norm_critical_latency +      # dominant latency signal for critical paths
        0.25 * energy_penalty +             # risk-gated energy cost
        0.20 * wait_penalty +               # anti-starvation
        0.10 * norm_slack_sensitivity +     # direct urgency intensity
        0.10 * work_density_penalty         # efficiency-aware load balancing
    )
    
    # Final bounds & cleanup
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
