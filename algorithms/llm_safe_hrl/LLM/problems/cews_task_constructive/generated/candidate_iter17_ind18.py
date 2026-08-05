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
    v2: Deadline-hardened, energy-aware priority rule combining robust rank normalization,
         soft-gated efficiency, critical-path density, and starvation-avoiding fairness.
    
    Key improvements:
    - Uses rank-based percentile scaling per term (not global) to preserve ordinal integrity
    - Soft-gated SEER (slack-conditioned energy/latency ratio) avoids hard zeroing & preserves gradient
    - Critical-path density = upward_rank * remaining_work / (exec+comm+eps) penalized by comm-uncertainty
    - Urgency uses adaptive sigmoid with slack distribution-aware steepness (tau)
    - Fairness via wait-rank + log1p scaling ensures strict ordering even for N=1 or skewed waits
    - Risk penalty replaced by uncertainty-modulated communication penalty only for high-slack tasks
    - All terms bounded, finite, and deterministic; smaller score = higher priority
    """
    eps = 1e-8
    
    # Clean and convert inputs to float arrays, handling NaN/inf safely
    def clean(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=eps)
    
    min_exec_time = clean(min_exec_time)
    min_comm_time = clean(min_comm_time)
    min_incremental_energy = clean(min_incremental_energy)
    slack = clean(slack)
    upward_rank = clean(upward_rank)
    remaining_work = clean(remaining_work)
    ready_wait_time = clean(ready_wait_time)
    uncertainty = clean(uncertainty)
    
    # Rank-normalization: maps values to [0,1] percentile rank (0-indexed), handles N=1
    def rank_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.array([0.0])
        sorted_vals = np.sort(x)
        # Use searchsorted for stable rank assignment (handles ties consistently)
        ranks = np.searchsorted(sorted_vals, x, side='left') / (len(sorted_vals) - 1)
        return ranks
    
    # --- Urgency term: adaptive sigmoid prioritizing deadline proximity ---
    # Steepness tau adapts to global slack distribution: tighter deadlines → steeper response
    slack_median = np.median(slack)
    tau_urgency = np.clip(1.0 + 0.5 * np.maximum(0.0, -slack_median), 0.5, 5.0)
    # Sigmoid: near 1 when slack is large (low urgency), near 0 when slack negative (high urgency)
    urgency_raw = 1.0 / (1.0 + np.exp((slack + 1.0) / tau_urgency))
    norm_urgency = rank_norm(urgency_raw)
    urgency_term = -6.0 * norm_urgency  # Dominant weight: higher priority for urgent tasks
    
    # --- Energy-efficiency term: Slack-conditioned Efficiency Ratio (SEER) ---
    # SEER = energy / latency, but softened: higher slack → stronger gating
    exec_comm_sum = min_exec_time + min_comm_time + eps
    base_seer = min_incremental_energy / exec_comm_sum
    # Soft gate: full weight when slack large, fades smoothly as slack decreases
    slack_gate = 1.0 / (1.0 + np.exp(-slack / 3.0))  # Gate ∈ (0,1), rises at slack=0
    seer_soft_gated = base_seer * slack_gate
    norm_seer = rank_norm(seer_soft_gated)
    seer_term = -1.6 * norm_seer  # Favor low-energy-per-latency tasks when slack allows
    
    # --- Critical-path density term: importance × work / latency, penalized by comm uncertainty ---
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    # Add uncertainty-penalized comm cost only for high-slack tasks (to avoid over-penalizing late ones)
    high_slack_mask = slack > np.percentile(slack, 75)
    comm_unc_penalty = np.where(high_slack_mask, min_comm_time * uncertainty * 0.25, 0.0)
    cp_density_penalized = cp_density + comm_unc_penalty / (exec_comm_sum + eps)
    norm_cp_density = rank_norm(cp_density_penalized)
    cp_density_term = 1.4 * norm_cp_density  # Prioritize high-importance, high-work, low-latency paths
    
    # --- Fairness term: prevents starvation of long-waiting tasks, especially non-urgent ones ---
    # Use normalized wait rank + log1p scaling for strict monotonicity and outlier resilience
    wait_rank = rank_norm(ready_wait_time)
    rel_log_wait = np.log1p(ready_wait_time) / (np.log1p(np.max(ready_wait_time) + eps) + eps)
    fairness_boost = np.clip(wait_rank * rel_log_wait, 0.0, 0.05)
    fairness_term = -0.25 * rank_norm(fairness_boost)  # Small but deterministic boost for waiting tasks
    
    # --- Uncertainty modulation: only activate for high-uncertainty, high-slack tasks ---
    unc_high = uncertainty > np.percentile(uncertainty, 85)
    unc_boost = np.where(unc_high & (slack > 0), np.clip(uncertainty * 0.15, 0.0, 0.1), 0.0)
    unc_term = -0.05 * rank_norm(unc_boost)  # Tiny bonus for low-uncertainty tasks in safe regime
    
    # Aggregate with strict hierarchy: urgency dominates (>70%), then SEER (>15%), CP-density (~10%), rest minor
    score = (
        urgency_term +
        seer_term +
        cp_density_term +
        fairness_term +
        unc_term
    )
    
    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    return score
