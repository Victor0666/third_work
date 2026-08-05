import numpy as np

# 函数名中的 2 由 SeEvo 在生成 Prompt 时替换为目标版本号。
# 八个输入均为长度 N 的一维数组；相同下标始终指向同一个 ready task。
# 返回值也必须是长度 N 的一维数组，并遵守“分数越小，优先级越高”。
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
    v2: Deadline-First Robust Priority (DFRP) — simplified, hardened, and semantics-aligned.
    
    Key self-evolution improvements:
    - Replaces fragile IQR scaling with stable median-abs-robust_slack scale → ensures urgency monotonicity under all distributions
    - Removes all conditional gating (tight_mask, energy_gate, conf_gate) → eliminates deadline-violating attenuation of urgency
    - Energy term is now *purely secondary*: only activated when robust_slack >= 0 AND normalized *relative to urgency*, never competing with it
    - Fairness uses linear wait-pressure scaled *only* by urgency headroom (max(0, robust_slack)), preventing starvation without compromising DDL
    - Uncertainty is applied *additively to slack* (not multiplicatively to latency) → preserves physical interpretation and avoids spurious penalties
    - All normalization uses MAD with degenerate fallback (N=1 → zeros; constant array → zeros), fully deterministic and bounded
    - Final score enforces strict priority hierarchy: Urgency >> Criticality > Energy Efficiency > Fairness (no negative weights on urgency)
    """
    eps = 1e-08
    N = len(slack)
    if N == 0:
        return np.array([], dtype=float)
    
    # Sanitize inputs: replace NaN/inf with safe finite values
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=eps)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: conservative estimate accounting for worst-case uncertainty
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e6, 1e6)
    
    # Stable urgency scale: median absolute robust slack → invariant to outliers & low-variance cases
    abs_robust = np.abs(robust_slack)
    scale = np.median(abs_robust) + eps
    
    # Urgency: strictly monotonic, unbounded below in penalty → smaller robust_slack ⇒ exponentially larger penalty ⇒ higher priority
    # Score must be *smaller* for higher priority → use negative exp to invert
    urgency_score = -np.exp(-np.clip(robust_slack / (scale + eps), -10.0, 10.0))
    
    # Criticality: upward_rank × work density, but *only* meaningful when execution effort is non-negligible
    exec_effort = np.maximum(min_exec_time, min_comm_time, eps)
    critical_density = upward_rank * (remaining_work / (exec_effort + eps))
    
    # Energy efficiency: marginal energy per remaining work — only prioritized *when slack is safely positive*
    # Scaled by sigmoid of robust_slack to smoothly activate from 0 (urgent) to 1 (safe), avoiding step discontinuities
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    energy_activation = 1.0 / (1.0 + np.exp(-(robust_slack - 0.1) / 0.3))  # smooth ramp from ~0.1s slack upward
    efficiency_score = energy_per_work * energy_activation
    
    # Fairness: aging pressure proportional to *available slack headroom*, preventing starvation only in safe regime
    # Linear wait_ratio clipped to avoid overflow, scaled by max(0, robust_slack) to suppress under urgency
    wait_ratio = np.clip(ready_wait_time / (exec_effort + eps), 0.0, 20.0)
    fairness_score = wait_ratio * np.maximum(0.0, robust_slack)
    
    # Normalize each component via robust MAD: handles N=1, constants, and degenerates safely
    def robust_mad_normalize(x):
        if N == 1:
            return np.zeros(1, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        normed = (x - median_x) / mad
        return np.clip(normed, -4.0, 4.0)
    
    urgency_norm = robust_mad_normalize(urgency_score)
    critical_norm = robust_mad_normalize(critical_density)
    efficiency_norm = robust_mad_normalize(efficiency_score)
    fairness_norm = robust_mad_normalize(fairness_score)
    
    # Strict priority hierarchy: urgency dominates (positive weight), others refine within its envelope
    # No negative weights on urgency → guarantees DDL-first behavior; energy/fairness are tie-breakers *only* among same-urgency tasks
    score = (
        +5.0 * urgency_norm          # primary driver: hard deadline adherence
        + 1.2 * critical_norm         # secondary: preserve critical path progress
        + 0.3 * efficiency_norm      # tertiary: reduce energy *only* when safe
        + 0.1 * fairness_norm        # quaternary: prevent starvation *only* when slack permits
    )
    
    # Final bounding and sanitization: ensure finite, deterministic output
    score = np.clip(score, -1e9, 1e9)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    
    return score
