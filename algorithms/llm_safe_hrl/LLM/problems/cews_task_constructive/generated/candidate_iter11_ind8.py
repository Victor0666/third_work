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
    Hybrid priority rule v2: hard-deadline enforcement + risk-aware energy efficiency + starvation-resilient fairness.
    
    Key design principles:
    - Strict DDL compliance: any negative slack → maximum urgency (Parent 2 strength)
    - Clean separation: urgency/lateness (hard constraint) vs energy/criticality (soft objective)
    - Adaptive uncertainty gating: only amplify uncertainty under tight slack (rel_slack <= 0.1) to avoid over-penalization
    - Robust work-normalized starvation boost: activated only when both slack > -0.02 AND wait_ratio in top 10%
    - Criticality-energy term uses upward_rank-weighted energy_per_work, normalized via robust_minmax for stability
    - All operations protected against NaN/inf/div0; deterministic and finite output
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

    def robust_minmax(x):
        if x.size == 0:
            return np.zeros_like(x)
        x_min, x_max = np.min(x), np.max(x)
        rng = x_max - x_min + eps
        return np.clip((x - x_min) / rng, 0.0, 1.0)

    # Task duration baseline
    task_duration = min_exec_time + min_comm_time + eps

    # --- HARD DEADLINE ENFORCEMENT (Parent 2 core strength) ---
    has_negative_slack = (slack < 0.0).astype(float)
    # Absolute lateness penalty scaled by median task duration
    lateness_penalty = np.where(slack < 0.0, np.clip(-slack / (np.median(task_duration) + eps), 0.0, 4.0), 0.0)
    # Urgency: binary override for violation + sigmoid for near-deadline pressure
    rel_slack = slack / task_duration
    urgency_sigmoid = 1.0 / (1.0 + np.exp(6.0 * (rel_slack - np.quantile(rel_slack, 0.3, method='midpoint') if N > 1 else 0.0)))
    urgency = np.where(has_negative_slack, 1.0, urgency_sigmoid)

    # --- RISK-AWARE ENERGY TERM (Hybrid: Parent 2 structure + Parent 1 normalization insight) ---
    energy_per_work = np.maximum(min_incremental_energy / (remaining_work + eps), eps)
    # Weight by criticality: higher upward_rank → higher priority for energy-efficient critical tasks
    crit_weighted_energy = energy_per_work * (1.0 + upward_rank / (np.median(upward_rank + eps) + eps))
    # Normalize to [0,1] for stable weighting
    norm_energy = robust_minmax(crit_weighted_energy)

    # --- UNCERTAINTY AMPLIFICATION (Parent 2 gating + Parent 1 risk-exponent nuance) ---
    tight_slack_mask = (rel_slack <= 0.1).astype(float)
    # Only apply uncertainty boost under tight slack; use normalized dur_uncertainty
    dur_uncertainty = np.where(task_duration > eps, uncertainty / task_duration, 0.0)
    uncertainty_boost = dur_uncertainty * tight_slack_mask
    norm_uncertainty = robust_minmax(uncertainty_boost)

    # --- STARVATION RESILIENCE (Parent 2 gating + Parent 1 work-relative insight) ---
    wait_per_work = ready_wait_time / (remaining_work + eps)
    p90_wait_ratio = np.percentile(wait_per_work, 90, method='midpoint') if N > 1 else np.max(wait_per_work)
    # Gate: only boost if slack is safe AND work is non-trivial AND wait is extreme
    wait_gate = (
        (rel_slack > -0.02).astype(float) *
        (remaining_work > np.quantile(remaining_work, 0.2, method='midpoint') + eps).astype(float) *
        (wait_per_work > p90_wait_ratio + eps).astype(float)
    )
    norm_wait_time = robust_minmax(ready_wait_time)
    wait_boost = norm_wait_time * wait_gate * 0.25

    # --- CRITICALITY NORMALIZATION (Parent 2 robustness) ---
    norm_upward = robust_minmax(upward_rank)

    # --- FINAL SCORE: smaller = better ---
    # Prioritize urgency (negative weight), penalize lateness & energy & uncertainty, reward criticality & fairness
    norm_urgency = robust_minmax(urgency)
    norm_lateness = robust_minmax(lateness_penalty)
    
    score = (
        -3.0 * norm_urgency           # Highest priority: meet deadline
        + 1.3 * norm_lateness         # Penalty for actual violation
        + 1.0 * norm_energy           # Energy minimization (soft objective)
        + 0.3 * norm_uncertainty      # Risk amplification only when needed
        - 0.4 * norm_upward           # Prefer higher criticality (lower score)
        + 0.15 * wait_boost           # Starvation relief when safe
    )

    # Final safeguard: ensure finite deterministic output
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
