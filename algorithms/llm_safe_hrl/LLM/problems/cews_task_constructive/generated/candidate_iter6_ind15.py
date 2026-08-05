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
    Self-evolved priority rule: refines deadline urgency, strengthens energy-efficiency grounding,
    introduces *dynamic critical-path awareness*, and adds *robustness-aware fairness*.
    
    Key advances over v1:
    - Replaces static arctan deadline risk with *adaptive sigmoid risk*: steeper near zero slack,
      flatter for large positive/negative slack → better sensitivity to critical margins.
    - Critical-energy density now incorporates *slack-aware weighting*: prioritizes high-impact work
      *only when slack permits*; de-emphasizes energy savings under violation (slack < 0).
    - Introduces *dynamic critical-path boost*: upward_rank scaled by normalized remaining_work
      relative to workflow-wide median → amplifies tasks on long/complex paths without double-counting.
    - Fairness boost now uses *relative latency burden* (vs. global max) *and* penalizes idle VMs implicitly
      via ready_wait_time interaction → improves load balancing under heterogeneity.
    - Uncertainty gating tightened: only active when slack ∈ (0, 3.0] and uncertainty > 0.05 → focuses on
      high-risk, low-margin regimes where estimation error matters most.
    - All normalizations use MAD with explicit nan/inf sanitization before normalization.
    - Final score bounded and sanitized with strict finite-only guarantee.
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
    
    # Sanitize inputs against NaN/inf early
    def sanitize(x):
        return np.nan_to_num(x, nan=eps, posinf=1e12, neginf=eps)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    def safe_mad_normalize(x):
        """Robust MAD normalization: handles N=1, constant arrays, and outliers."""
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - median_x) / mad
        return np.clip(normed, -3.0, 3.0)
    
    # Adaptive deadline risk: steep sigmoid near slack=0, bounded [0,1]
    # Avoids flat regions of arctan; more discriminative for critical deadlines
    deadline_risk = 1.0 / (1.0 + np.exp(-slack / (np.abs(np.median(slack)) + eps)))
    deadline_score = safe_mad_normalize(deadline_risk)
    
    # Slack-conditioned critical-energy density: suppress energy focus when violating
    energy_denom = min_incremental_energy + eps
    base_density = upward_rank * remaining_work / energy_denom
    # Only reward energy efficiency when slack > 0; otherwise neutral (0)
    critical_energy_density = np.where(slack > 0.0, base_density, 0.0)
    critical_energy_norm = safe_mad_normalize(critical_energy_density)
    
    # Dynamic critical-path boost: upward_rank × (remaining_work / median_remaining_work)
    # Highlights tasks on long descendant chains, avoids bias toward shallow high-rank nodes
    median_rw = np.median(remaining_work) + eps
    cp_boost = upward_rank * (remaining_work / median_rw)
    cp_boost_norm = safe_mad_normalize(cp_boost)
    
    # Latency fairness: proportional to relative latency burden, enhanced by wait-time urgency
    total_latency = min_exec_time + min_comm_time + eps
    max_latency = np.max(total_latency) + eps
    latency_ratio = total_latency / max_latency
    # Boost fairness for long-latency tasks *that have waited*, capped at 0.15
    fairness_boost = np.clip(latency_ratio * (1.0 + 0.5 * np.tanh(ready_wait_time / (np.abs(slack) + 1.0))), 0.0, 0.15)
    
    # Tighter uncertainty gating: only active in high-risk, low-margin zone
    # slack ∈ (0, 3.0] and uncertainty > 0.05 → avoids noise in safe or deeply violated regimes
    uncertainty_gated = np.where(
        (slack > 0.0) & (slack <= 3.0) & (uncertainty > 0.05),
        uncertainty * (1.0 / (1.0 + np.exp(-(2.0 - slack)))),
        0.0
    )
    uncertainty_norm = safe_mad_normalize(uncertainty_gated)
    
    # Aging boost refined: only activates when slack < 2.0s (tighter threshold) and wait time > 0.1s
    aging_boost = np.where(
        (slack < 2.0) & (ready_wait_time > 0.1),
        np.tanh(0.8 * ready_wait_time / (np.abs(slack) + 0.5)),
        0.0
    )
    
    # Weight hierarchy tuned: deadline urgency dominates, critical-path and energy guide efficiency,
    # fairness & aging ensure liveness, uncertainty refines margin decisions
    score = (
        +4.2 * deadline_score
        - 2.3 * critical_energy_norm
        - 1.3 * cp_boost_norm
        + 0.15 * fairness_boost
        + 0.09 * aging_boost
        + 0.16 * uncertainty_norm
    )
    
    # Strict final sanitization: ensure finite, bounded, shape-(N,)
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
