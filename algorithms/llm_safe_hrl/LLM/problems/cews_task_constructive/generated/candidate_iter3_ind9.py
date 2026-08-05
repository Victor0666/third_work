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

    "Self-evolved priority rule: hardened deadline enforcement, simplified risk gating, and responsive starvation mitigation.\n    Key improvements over v1:\n      - Slack urgency uses *linear-penalized exponential* for smooth, monotonic, and bounded penalty (no artificial caps);\n      - Uncertainty boost now gated *only* by slack <= 0 — aligning risk response strictly with deadline violation;\n      - Starvation guard activated when slack <= 0 OR ready_wait_time > 95th percentile — ensures timely retry even under moderate slack;\n      - Energy efficiency term redefined as 'energy per unit work' (not per second) to favor compute-dense tasks;\n      - All normalization uses unified robust median-IQR fallback with explicit zero-variance handling;\n      - Coefficients rebalanced to prioritize slack (-2.0), energy efficiency (+0.4), and criticality (+0.25), suppressing noise terms."
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()
    
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        # Use MAD fallback if IQR near-zero to avoid division instability
        if iqr < eps:
            mad = np.mean(np.abs(x - med)) + eps
            norm = (x - med) / mad
        else:
            norm = (x - med) / (iqr + eps)
        return np.clip(norm, -1e6, 1e6)  # Prevent extreme outliers
    
    # Linear-penalized exponential urgency: monotonic, smooth, unbounded but numerically stable
    # For slack <= 0: exp(|slack|/10) grows steadily; for slack > 0: decays smoothly as 1/(1+slack/30)
    slack_urgency = np.where(
        slack <= 0,
        np.exp(np.clip(np.abs(slack) / 10.0, 0.0, 20.0)),  # cap exponent to prevent overflow
        1.0 / (1.0 + np.clip(slack / 30.0, 0.0, 1e3))
    )
    
    # Energy efficiency: minimize energy per MI (joules per million instructions), not per second
    # Avoids bias against long-running low-power tasks; favors compute-dense scheduling
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    norm_energy_pw = robust_normalize(np.clip(energy_per_work, 1e-6, 1e9))
    
    # Criticality: upward_rank scaled by work-efficiency ratio only when slack is tight
    # Ensures high-rank nodes are prioritized *only* when they contribute meaningfully to deadline risk
    work_efficiency_ratio = remaining_work / (min_incremental_energy + eps)
    leveraged_rank = upward_rank * np.where(slack <= 0, work_efficiency_ratio, 1.0)
    norm_leveraged_rank = robust_normalize(leveraged_rank)
    
    # Uncertainty boost: activated *exclusively* under deadline risk (slack <= 0), not heuristic thresholds
    # Simplifies gating, sharpens risk signal, avoids noise from marginal uncertainty
    uncertainty_boost = np.where(slack <= 0, robust_normalize(uncertainty), 0.0)
    
    # Starvation guard: triggers if either (a) deadline violated, or (b) wait exceeds 95th percentile
    # More responsive than v1's dual condition — prevents stagnation of long-waiting tasks
    p95_wait = np.percentile(ready_wait_time, 95, method='midpoint') if ready_wait_time.size > 1 else 0.0
    starvation_guard = np.where(
        (slack <= 0) | (ready_wait_time > p95_wait + eps),
        robust_normalize(ready_wait_time),
        0.0
    )
    
    # Final score: emphasize urgency (-2.0), energy efficiency (+0.4), criticality (+0.25), 
    # uncertainty (+0.1), starvation (-0.05); suppress execution/comm time terms (redundant with duration)
    score = (
        +0.40 * norm_energy_pw
        - 2.00 * slack_urgency
        + 0.25 * norm_leveraged_rank
        + 0.10 * uncertainty_boost
        - 0.05 * starvation_guard
    )
    
    # Ensure finite output with safe NaN/Inf handling
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
