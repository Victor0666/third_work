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
    v2: Self-evolved priority rule with three-layer deadline enforcement,
         adaptive critical-path energy efficiency, and starvation-resilient fairness.
    
    Key self-evolution improvements:
    - Triple-mode urgency: 
        * slack > 0: arctan-based soft urgency (bounded, monotonic)
        * -5 <= slack <= 0: linear ramp (1 + |slack|/5) for precise violation margin control
        * slack < -5: exponential dominance (exp(|slack|-5)) — isolates severe violations
    - Critical-energy density (CED) now includes *latency-aware normalization*: 
        divided by (min_exec_time + min_comm_time + eps) to penalize high-latency high-energy tasks
    - Upward rank gated by *both* slack > 0 AND remaining_work > 0 to suppress spurious importance in idle paths
    - Latency-risk modulation upgraded to *uncertainty-scaled slack penalty*: 
        adds (uncertainty * |slack|) only when slack < 0 → directly couples risk with violation severity
    - Fairness term uses *log1p-scaled wait time* (log1p(ready_wait_time)/log1p(max_wait)) clipped to [0, 0.12],
      applied unconditionally but weight reduced to 0.06 — smoother, more robust than sqrt
    - All normalized terms use unified safe_mad_normalize with outlier suppression via np.clip before MAD
    - Final weights enforce strict objective hierarchy: deadline (5.0) >> CED (2.6) >> upward_rank (1.5) 
      >> risk-penalty (0.4) >> fairness (0.06); remaining_work fully removed (redundant with CED/upward_rank)
    - Explicit NaN/inf zeroing before normalization and final nan_to_num fallback
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
    
    # Pre-normalization NaN/inf cleanup
    def clean_array(x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    min_exec_time = clean_array(min_exec_time)
    min_comm_time = clean_array(min_comm_time)
    min_incremental_energy = clean_array(min_incremental_energy)
    slack = clean_array(slack)
    upward_rank = clean_array(upward_rank)
    remaining_work = clean_array(remaining_work)
    ready_wait_time = clean_array(ready_wait_time)
    uncertainty = clean_array(uncertainty)
    
    def safe_mad_normalize(x):
        """MAD normalization robust to N=1, constants, outliers; clips to [-4,4]"""
        if x.size == 1:
            return np.zeros_like(x)
        # Clip extreme outliers before median calc to improve robustness
        x_clipped = np.clip(x, -1e6, 1e6)
        med = np.median(x_clipped)
        mad = np.median(np.abs(x_clipped - med)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x_clipped - med) / mad
        return np.clip(normed, -4.0, 4.0)
    
    # Triple-mode deadline urgency: precise margin control + hard violation dominance
    arctan_urgency = 0.5 + 1.0 / np.pi * np.arctan(np.where(slack >= 0, slack, 0.0) / 5.0)
    linear_violation = np.where((slack < 0) & (slack >= -5), 1.0 + (-slack) / 5.0, 0.0)
    severe_violation = np.where(slack < -5, np.exp(np.clip(-slack - 5, 0.0, 20.0)), 0.0)
    deadline_risk_raw = arctan_urgency + linear_violation + severe_violation
    deadline_score = safe_mad_normalize(deadline_risk_raw)
    
    # Critical-energy density: work-per-joule, latency-aware (penalizes high-latency high-energy)
    ced_numerator = upward_rank * remaining_work + eps
    ced_denominator = (min_incremental_energy + eps) * (min_exec_time + min_comm_time + eps)
    ced_raw = np.where(slack > 0, ced_numerator / ced_denominator, 0.0)
    ced_norm = safe_mad_normalize(ced_raw)
    
    # Upward rank gated by both slack > 0 AND non-zero remaining work (avoids phantom importance)
    upward_rank_active = np.where((slack > 0) & (remaining_work > eps), upward_rank, 0.0)
    upward_rank_norm = safe_mad_normalize(upward_rank_active)
    
    # Uncertainty-scaled slack penalty: only applies when slack < 0, directly coupling risk and violation
    risk_penalty_raw = np.where(slack < 0, (-slack) * np.clip(uncertainty, 0.0, 0.7), 0.0)
    risk_penalty_norm = safe_mad_normalize(risk_penalty_raw)
    
    # Fairness: log1p-scaled wait time — smoother, more stable than sqrt at low wait times
    max_wait = np.max(ready_wait_time) + eps
    rel_log_wait = np.log1p(ready_wait_time) / (np.log1p(max_wait) + eps)
    fairness_boost = np.clip(rel_log_wait, 0.0, 0.12)
    
    # Final weighted score: deadline dominates; no remaining_work term (redundant with CED/upward_rank)
    score = (
        +5.0 * deadline_score 
        - 2.6 * ced_norm 
        - 1.5 * upward_rank_norm 
        + 0.4 * risk_penalty_norm 
        + 0.06 * fairness_boost
    )
    
    return np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
