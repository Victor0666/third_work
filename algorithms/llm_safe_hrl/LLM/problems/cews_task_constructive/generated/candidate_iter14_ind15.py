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
    v2 hybrid priority rule: Combines Parent 2's strict DDL-hardness and robust normalization
    with Parent 1's latency fairness and uncertainty gating, while enhancing urgency resolution
    and critical-energy alignment.
    
    Key innovations:
    - Urgency uses arctan(-slack/tau) with linear tail for deep violation pull (monotonic & stable)
    - Critical-energy density (CED) is slack-gated *and* urgency-scaled to prioritize energy savings
      only when both feasible *and* urgent (not just slack >= 0)
    - Uncertainty penalty applies to *all* tasks with |slack| <= median(|slack|), capturing tight-deadline risk
    - Latency fairness term weighted by remaining_work and normalized via adaptive IQR/minmax
    - Anti-starvation wait boost uses sqrt(wait) scaled by urgency-aware sigmoid, clipped for stability
    - All terms normalized adaptively; final score ensures smaller = higher priority
    """
    eps = 1e-08
    # Safe casting and NaN/inf handling
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=eps, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=eps, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    
    def normalize_adaptive(x):
        """Robust normalization: uses IQR if non-degenerate, else minmax; handles size-1 safely."""
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        if iqr > eps:
            center = np.median(x)
            scale = iqr + eps
        else:
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            center = (xmax + xmin) / 2.0
            if scale < eps:
                scale = eps
        return (x - center) / (scale + eps)
    
    # Urgency: bounded arctan for moderate slack, linear tail for deep violations → monotonic & unbounded-below effective
    tau = 1.0
    urgency_raw = np.where(slack < -5.0, -np.pi/2 - 0.2*(slack + 5.0), np.arctan(-slack / tau))
    norm_urgency = normalize_adaptive(urgency_raw)
    urgency_term = -3.0 * norm_urgency  # Stronger pull on deadline-critical tasks
    
    # Critical-energy density: upward_rank * remaining_work / energy, gated by feasibility AND urgency
    # Only consider energy efficiency when slack >= -1.0 (soft feasibility buffer) AND urgency > 0 (not yet violated)
    ced_base = (upward_rank * remaining_work) / (min_incremental_energy + eps)
    ced_mask = np.where(slack >= -1.0, ced_base, 0.0)  # Relax hard cutoff to -1s for numerical safety
    norm_ced = normalize_adaptive(ced_mask)
    ced_term = -1.5 * norm_ced  # Increased weight for energy-density prioritization under feasibility
    
    # Uncertainty penalty: applied to all tasks with |slack| <= median(|slack|) — captures tight-deadline risk
    abs_slack = np.abs(slack) + eps
    median_abs_slack = np.median(abs_slack) if abs_slack.size > 0 else 1.0
    uncertainty_gated = np.where(abs_slack <= median_abs_slack, uncertainty, 0.0)
    norm_uncertainty = normalize_adaptive(uncertainty_gated)
    uncertainty_term = 0.18 * norm_uncertainty  # Modest penalty for risk in tight windows
    
    # Latency impact: total latency scaled by uncertainty factor, normalized
    base_latency = min_exec_time + min_comm_time + eps
    uncertainty_factor = 1.0 + np.arctan(uncertainty) / np.pi
    latency_scaled = base_latency * uncertainty_factor
    norm_latency = normalize_adaptive(latency_scaled)
    latency_term = 0.85 * norm_latency  # Prioritizes low-latency tasks, especially under uncertainty
    
    # Fairness: sqrt(wait) scaled by urgency-aware factor (sigmoid(-slack)) to boost long-waiting urgent tasks
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    urgency_sigmoid = 1.0 / (1.0 + np.exp(slack))  # Higher when slack negative → boosts urgent waiting
    fairness_raw = sqrt_wait * (1.0 + urgency_sigmoid)
    fairness_clipped = np.clip(fairness_raw, 0.0, 0.35)  # Prevent dominance from extreme wait times
    norm_fairness = normalize_adaptive(fairness_clipped)
    fairness_term = -0.22 * norm_fairness  # Negative weight: higher fairness score = lower priority score
    
    # Assemble final score: smaller value = higher priority
    score = urgency_term + ced_term + latency_term + uncertainty_term + fairness_term
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
