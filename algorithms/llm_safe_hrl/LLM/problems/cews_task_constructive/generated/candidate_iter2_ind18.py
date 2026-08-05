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
    'Improved priority rule combining deadline-safety gating, robust energy-latency efficiency,\n    risk-weighted criticality, and calibrated starvation mitigation.\n    \n    Key improvements:\n    - Uses *dual-gated criticality*: upward_rank contributes fully when slack > 0,\n      linearly decays to zero at slack = -tau, and stays zero beyond (smoother than hard gate).\n    - Introduces *uncertainty-aware energy-latency ratio*: scales energy_eff_ratio by (1 + uncertainty)\n      to penalize high-risk efficient placements.\n    - Replaces percentile wait boost with *normalized wait saturation*: uses clipped sigmoid\n      on relative wait (robust to outliers, bounded [0,1]).\n    - Adds *slack-normalized urgency penalty*: transforms slack into smooth, bounded urgency signal\n      that dominates score when deadlines are tight or violated.\n    - All normalizations use IQR-based robust scaling with strict clipping [-3, 3].\n    - Final score is convex combination emphasizing deadline safety first, then efficiency and criticality.\n    '
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
    
    # Compute adaptive deadline risk: exponential penalty for negative slack, bounded
    tau = np.clip(np.mean(np.abs(slack)) + eps, eps, 1000.0)
    urgency_penalty = np.where(
        slack < 0,
        np.exp(np.clip(-slack / tau, 0.0, 20.0)) - 1.0,
        0.0
    )
    
    # Dual-gated upward_rank: full contribution when slack > 0, linear decay to zero at slack = -tau, zero beyond
    rank_gate = np.clip((slack + tau) / tau, 0.0, 1.0)
    upward_rank_gated = upward_rank * rank_gate
    
    # Energy efficiency: energy per total latency, amplified by uncertainty (higher risk → lower efficiency priority)
    total_latency = min_exec_time + min_comm_time + eps
    energy_eff_ratio = min_incremental_energy / total_latency
    energy_eff_risk_adjusted = energy_eff_ratio * (1.0 + uncertainty)
    
    # Starvation mitigation: normalized wait saturation via clipped sigmoid on relative wait
    if len(ready_wait_time) == 1:
        wait_norm = np.array([0.0])
    else:
        wait_mean = np.mean(ready_wait_time)
        wait_std = np.std(ready_wait_time) + eps
        wait_zscore = (ready_wait_time - wait_mean) / wait_std
        # Saturate at reasonable bounds to avoid extreme values
        wait_zscore_clipped = np.clip(wait_zscore, -5.0, 5.0)
        wait_norm = 1.0 / (1.0 + np.exp(-0.5 * wait_zscore_clipped))
    
    # Normalize all components
    urgency_norm = robust_normalize(urgency_penalty)
    rank_norm = robust_normalize(upward_rank_gated)
    energy_norm = robust_normalize(energy_eff_risk_adjusted)
    work_norm = robust_normalize(remaining_work)
    wait_norm_final = robust_normalize(wait_norm)
    
    # Final score: prioritize deadline safety first, then efficiency, critical path, and starvation control
    # Coefficients sum to ~6.0 — balanced emphasis without dominance
    score = (
        3.0 * urgency_norm +          # Highest weight: hard deadline enforcement
        1.5 * energy_norm +          # Strong weight: energy efficiency under risk
        1.0 * rank_norm +            # Moderate weight: critical-path progress when safe
        0.3 * work_norm +            # Light weight: total workload as secondary proxy for impact
        0.2 * wait_norm_final         # Minimal but nonzero: prevents indefinite starvation
    )
    
    # Ensure finite output
    return np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
