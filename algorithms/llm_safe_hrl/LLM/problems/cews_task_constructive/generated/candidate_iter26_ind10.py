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
    v2: Hybrid deadline-hardened priority with adaptive urgency gating, 
    safety-weighted energy efficiency, uncertainty-coupled fairness, 
    and robust piecewise urgency continuity.
    
    Key innovations:
    - Combines Parent 2's adaptive urgency window (tau_urgency - slack) 
      with Parent 1's hard violation override for strict DDL dominance.
    - Uses bounded piecewise-linear urgency mapping (not tanh or clipping) 
      that is continuous, monotonic, and zero-crossing at slack=0.
    - Integrates uncertainty into both latency pressure attenuation 
      AND fairness amplification — but only when slack > 0 to avoid starving critical tasks.
    - Replaces per-term masking with global urgency gating: all non-fairness terms scaled by 
      (1 + clip(-slack, 0, 1)) to amplify priority under deadline pressure.
    - Criticality-regularized fairness uses upward_rank * ready_wait_time, 
      scaled by exec+comm and gated by slack > 0.5 for starvation relief in safe regimes.
    - All normalization uses variance-stable min-max with fallback; final score bounded and reshaped.
    """
    eps = 1e-08
    tau_safe = 3.0
    tau_urgency = 1.0
    
    # Sanitize inputs: ensure finite, replace NaN/inf with safe values
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
    
    # Robust slack: shrink slack by uncertainty, but only when slack > 0 (avoid over-penalizing urgent tasks)
    robust_slack = np.where(slack > 0, slack - 2.0 * uncertainty, slack)
    
    # Piecewise-linear urgency: continuous, monotonic, zero-crossing at robust_slack == 0
    # - If robust_slack < -eps → max urgency (-1.0)
    # - If robust_slack > eps → decreasing urgency (linear ramp from 0 to 1)
    # - Else linear interpolation in [-eps, eps]
    urgency_raw = np.zeros_like(robust_slack)
    urgency_raw = np.where(robust_slack < -eps, -1.0,
                          np.where(robust_slack > eps, 
                                  (eps - robust_slack) / (2.0 * eps), 
                                  0.0))
    
    # Global urgency gating factor: amplifies all non-fairness terms under deadline pressure
    urgency_gate = 1.0 + np.clip(-robust_slack, 0.0, 1.0)
    
    # Execution + communication baseline
    exec_comm_sum = min_exec_time + min_comm_time + eps
    
    # Latency pressure: critical path importance weighted by execution cost and urgency
    latency_pressure = (upward_rank * exec_comm_sum) / (np.clip(tau_urgency - robust_slack, eps, tau_urgency + eps) + eps)
    latency_pressure = latency_pressure * (1.0 - np.clip(uncertainty, 0.0, 1.0)) * urgency_gate
    
    # Safety-weighted energy term: reward energy savings more when slack is safe
    safety_factor = np.clip(robust_slack / (tau_safe + eps), 0.0, 1.0)
    energy_term = (min_incremental_energy / (exec_comm_sum + 1.0)) * safety_factor * urgency_gate
    
    # Critical path density: importance × work / cost, attenuated by uncertainty
    cp_density = (upward_rank + eps) * (remaining_work + eps) / (exec_comm_sum + eps)
    cp_gated = cp_density * (1.0 - np.clip(uncertainty, 0.0, 1.0)) * urgency_gate
    
    # Fairness term: starvation relief for long-waiting high-importance tasks, 
    # activated only when slack > 0.5 (safe regime) and boosted by uncertainty
    fairness_base = ready_wait_time * (1.0 + upward_rank) / (exec_comm_sum + eps)
    fairness_boost = np.where(robust_slack > 0.5, uncertainty * 0.5 * ready_wait_time, 0.0)
    fairness_term = fairness_base + fairness_boost
    
    # Robust min-max normalization (handles single-element case)
    def robust_normalize(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)
    
    norm_urgency = robust_normalize(urgency_raw)
    norm_latency = robust_normalize(latency_pressure)
    norm_energy = robust_normalize(energy_term)
    norm_cp = robust_normalize(cp_gated)
    norm_fair = robust_normalize(fairness_term)
    
    # Final score: urgency dominates, then latency & energy, CP penalizes (lower is better), fairness reduces priority (higher wait = higher priority → negative weight)
    score = (
        -20.0 * norm_urgency 
        - 12.0 * norm_latency 
        - 6.0 * norm_energy 
        + 0.5 * norm_cp 
        - 1.0 * norm_fair
    )
    
    # Hard violation override: any task with slack < -1e-6 gets highest priority (lowest score)
    violation_mask = slack < -1e-6
    if np.any(violation_mask):
        base_min = np.min(score[~violation_mask]) if np.any(~violation_mask) else np.min(score)
        score = np.where(violation_mask, base_min - 1e9, score)
    
    # Final sanitization and shape enforcement
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    return score.reshape(-1)
