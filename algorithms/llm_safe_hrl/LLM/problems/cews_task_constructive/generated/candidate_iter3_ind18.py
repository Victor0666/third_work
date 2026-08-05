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

    'Self-evolved priority rule: deadline-safe criticality preservation, noise-robust urgency, and single-normalization efficiency.\n    \n    Key evolutions:\n    - Restores *critical-path dominance under pressure*: upward_rank scaled by soft-gated urgency (not suppressed) to preserve makespan-aware energy minimization.\n    - Replaces unstable exponential urgency with *bounded tanh-based urgency* — smooth, saturating, immune to slack estimation outliers.\n    - Eliminates double normalization: only raw features requiring scale alignment are normalized; bounded signals (e.g., wait saturation) skip robust_normalize.\n    - Introduces *energy-latency leverage*: ratio of (min_incremental_energy / (min_exec_time + min_comm_time + eps)) weighted by (1 + uncertainty), then capped to [0, 10] for stability.\n    - Adds *work-aware urgency scaling*: multiplies urgency signal by normalized remaining_work to prioritize high-impact late tasks.\n    - Uses *wait saturation via clipped linear ramp* (0→1 over top 25% wait times) — interpretable, monotonic, outlier-immune.\n    '
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    def robust_normalize(x):
        """IQR-based normalization: x -> (x - Q1) / (Q3 - Q1 + eps), clipped to [-3, 3]"""
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1 + eps
        normed = (x - q1) / iqr
        return np.clip(normed, -3.0, 3.0)
    
    # Bounded, noise-robust urgency: tanh(-slack / max(|slack|_mean, eps)) → [-1,1], then shifted & clipped to [0, 1]
    slack_abs_mean = np.mean(np.abs(slack)) + eps
    urgency_raw = np.tanh(-slack / slack_abs_mean)  # negative slack → positive urgency
    urgency = np.clip((urgency_raw + 1.0) / 2.0, 0.0, 1.0)  # map [-1,1] → [0,1]
    
    # Work-aware urgency amplification: prioritize late tasks with high remaining work
    work_norm = robust_normalize(remaining_work)
    urgency_weighted = urgency * np.clip(work_norm + 3.0, 0.1, 6.0)  # shift & bound to avoid zero-weight
    
    # Criticality preserved under pressure: upward_rank scaled by urgency (not gated out), then normalized
    criticality = upward_rank * (1.0 + urgency)  # boosts criticality when urgency > 0, no suppression
    criticality_norm = robust_normalize(criticality)
    
    # Stable energy-latency leverage: energy per unit time, risk-adjusted, bounded
    total_latency = min_exec_time + min_comm_time + eps
    energy_leverage = min_incremental_energy / total_latency
    energy_leverage_risk = energy_leverage * (1.0 + uncertainty)
    energy_leverage_bounded = np.clip(energy_leverage_risk, 0.0, 10.0)
    energy_norm = robust_normalize(energy_leverage_bounded)
    
    # Wait saturation: linear ramp from 0 to 1 over top 25% wait times (robust, monotonic, no sigmoid artifacts)
    if len(ready_wait_time) == 1:
        wait_saturation = np.array([0.0])
    else:
        wait_q75 = np.percentile(ready_wait_time, 75)
        wait_range = np.maximum(np.percentile(ready_wait_time, 100) - wait_q75, eps)
        wait_saturation = np.clip((ready_wait_time - wait_q75) / wait_range, 0.0, 1.0)
    
    # Final convex combination: urgency dominates, criticality anchors makespan, energy efficiency refines trade-off
    # No redundant normalization on wait_saturation or urgency_weighted — they are already bounded & meaningful
    score = (
        4.0 * urgency_weighted +
        2.0 * criticality_norm +
        1.2 * energy_norm +
        0.5 * wait_saturation +
        0.3 * robust_normalize(uncertainty)
    )
    
    return np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
