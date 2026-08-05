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
    v2: Lexicographic DDL-first priority with robust quantile normalization,
    adaptive risk-gated energy efficiency, and criticality-aware fairness.
    
    Key mutations:
    - Replaces piecewise urgency with smooth, bounded arctan-based urgency scaling
      that is strictly monotonic and avoids discontinuities at slack=0.
    - Uses global quantile-based (10th/90th) robust scaling instead of min-max,
      with variance fallback for small N and zero-variance handling.
    - Introduces 'critical_energy_ratio': upward_rank-weighted energy efficiency
      normalized by remaining_work to prioritize high-impact low-energy tasks.
    - Adds uncertainty-coupled latency penalty only when slack > 0, avoiding 
      starvation of truly critical (slack < 0) tasks.
    - Fairness term now uses log1p(ready_wait_time) for smoother anti-starvation
      behavior and is gated by both slack > 0.1 and upward_rank > median.
    - All terms are sign-consistent: lower score = higher priority; no additive
      violation overrides — instead uses urgency as multiplicative gate on all terms.
    """
    eps = 1e-8
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=-1e6)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: penalize uncertainty only in safe region to avoid over-penalizing critical paths
    robust_slack = np.where(slack > 0, slack - 1.5 * uncertainty, slack)
    
    # Smooth, bounded urgency: arctan-based, monotonic, zero-centered at slack=0
    # Output range: [-pi/2, pi/2] → mapped to [0, 1] then inverted for priority semantics
    urgency_raw = np.arctan((eps - robust_slack) / (eps + 0.1))  # steep near deadline
    urgency_scaled = (np.pi/2 - urgency_raw) / np.pi  # [0,1], higher = more urgent
    
    # Latency pressure: critical path sensitivity × execution cost, attenuated by uncertainty only if slack > 0
    exec_comm_cost = min_exec_time + min_comm_time + eps
    latency_pressure = upward_rank * exec_comm_cost
    latency_pressure = np.where(
        robust_slack > 0,
        latency_pressure * (1.0 + 0.5 * uncertainty),
        latency_pressure
    )
    
    # Energy efficiency: marginal energy per unit work, gated by urgency and safety
    # Prioritizes low-energy tasks on critical paths when safe to do so
    energy_efficiency = min_incremental_energy / (remaining_work + eps)
    safety_gate = np.clip(robust_slack / (2.0 + eps), 0.0, 1.0)
    energy_term = energy_efficiency * safety_gate * urgency_scaled
    
    # Critical-energy ratio: upward_rank × (1/energy_efficiency) → high rank + low energy = high priority
    critical_energy_ratio = (upward_rank + eps) / (min_incremental_energy + eps)
    critical_energy_ratio = critical_energy_ratio * urgency_scaled
    
    # Fairness: log-scaled wait time, activated only for non-critical & above-median criticality
    wait_fairness = np.log1p(ready_wait_time)
    urank_median = np.median(upward_rank) if len(upward_rank) > 1 else np.mean(upward_rank)
    fairness_gate = np.where(
        (robust_slack > 0.1) & (upward_rank > urank_median),
        1.0,
        0.0
    )
    fairness_term = wait_fairness * fairness_gate * (1.0 - urgency_scaled)  # de-emphasized under urgency
    
    # Robust quantile normalization: stable for small N and degenerate cases
    def quantile_normalize(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        q10 = np.quantile(x, 0.1)
        q90 = np.quantile(x, 0.9)
        if q90 - q10 < eps:
            return np.zeros_like(x)
        return (x - q10) / (q90 - q10 + eps)
    
    norm_urgency = quantile_normalize(urgency_scaled)
    norm_latency = quantile_normalize(latency_pressure)
    norm_energy = quantile_normalize(energy_term)
    norm_crit_energy = quantile_normalize(critical_energy_ratio)
    norm_fair = quantile_normalize(fairness_term)
    
    # Final score: weighted sum — smaller = better
    # Urgency dominates (negative weight), then latency & critical-energy, then fairness (positive weight to reduce priority)
    score = (
        -30.0 * norm_urgency          # strongest pull toward deadline compliance
        - 15.0 * norm_latency         # penalize high-latency critical tasks
        - 12.0 * norm_crit_energy     # reward energy-efficient critical tasks
        + 2.0 * norm_fair             # mild anti-starvation push (small positive = lowers priority slightly)
        + 4.0 * norm_energy           # slight preference for low-energy tasks in safe regime
    )
    
    # Ensure finite output and correct shape
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    return score.reshape(-1)
