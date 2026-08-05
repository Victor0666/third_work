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
    v3 self-evolved priority rule: Fixes v1's over-aggressive slack gating and weak urgency discrimination.
    
    Key improvements:
    - Replaces arctan with *clipped reciprocal urgency* (1/(slack + eps)) but safely bounded to [0, 1e6]
      → preserves high discriminability near zero slack without singularity explosion.
    - Relaxes slack gating: non-urgency terms now activate for *all slack >= 0*, not just slack >= median.
      Tight-but-feasible tasks (0 <= slack < median) retain energy/latency/fairness signals.
    - Separates uncertainty roles: 
        * latency_uncert = clipped(uncertainty) * (1 + max(0, median_slack_pos - slack)/median_slack_pos)
        * fairness_uncert = sqrt(uncertainty) * (1 + (1 - slack/margin_clip) if slack > 0 else 0)
      → avoids joint inflation; risk modulates latency more aggressively than fairness.
    - Introduces *deadline proximity ratio* (dpr = max(0, 1 - slack/median_slack_pos)) to smoothly scale
      urgency weight and uncertainty amplification — enabling graceful transition from safe to critical.
    - CED term uses *upward_rank / (min_incremental_energy + eps)* (inverse energy efficiency) weighted by dpr
      → prioritizes low-energy-high-criticality tasks under pressure.
    - All normalization uses MAD with strict N=1 handling, [-2.5, 2.5] clipping, and double nan_to_num.
    - Final score clamped and sanitized to guarantee finite deterministic output.
    """
    eps = 1e-8
    # Safe casting and NaN/inf sanitization
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    
    N = len(slack)
    if N == 0:
        return np.array([], dtype=float)
    
    def normalize_mad(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev)
        scale = mad if mad > eps else eps
        norm = (x - med) / scale
        return np.clip(norm, -2.5, 2.5)
    
    # Hard deadline violation override
    violation_mask = slack < 0
    hard_ddl_offset = np.where(violation_mask, -1000.0, 0.0)
    
    # Deadline proximity ratio: 0 (slack >> median) → 1 (slack ≤ 0), smooth ramp
    positive_slack_mask = slack > 0
    median_slack_pos = np.median(slack[positive_slack_mask]) if np.any(positive_slack_mask) else 1.0
    margin_clip = np.maximum(median_slack_pos, eps)
    dpr = np.clip(np.where(slack <= 0, 1.0, (margin_clip - slack) / margin_clip), 0.0, 1.0)
    
    # Urgency: clipped reciprocal with discriminative power near zero, bounded
    urgency_raw = np.where(violation_mask, -1e6, np.clip(1.0 / (slack + eps), 0.0, 1e6))
    norm_urgency = normalize_mad(urgency_raw)
    
    # CED term: criticality per unit energy — inverted for lower energy preference
    ced_base = upward_rank / (min_incremental_energy + eps)
    # Activate for all feasible (slack >= 0), scaled by deadline pressure
    ced_feasibility_gate = slack >= 0
    ced_masked = np.where(ced_feasibility_gate, ced_base * (1.0 + dpr), 0.0)
    norm_ced = normalize_mad(ced_masked)
    ced_term = -2.4 * norm_ced
    
    # Energy term: pure marginal energy minimization, gated only by feasibility
    energy_masked = np.where(ced_feasibility_gate, min_incremental_energy, 0.0)
    norm_energy = normalize_mad(energy_masked)
    energy_term = -1.4 * norm_energy
    
    # Latency term: base latency + uncertainty-modulated inflation
    base_latency = min_exec_time + min_comm_time + eps
    # Latency uncertainty amplification: stronger when slack tight
    latency_uncert = np.clip(uncertainty, 0.0, 1.0) * (1.0 + dpr)
    inflated_latency = base_latency * (1.0 + latency_uncert)
    norm_latency = normalize_mad(inflated_latency)
    latency_term = 0.8 * norm_latency
    
    # Fairness term: sqrt(wait) scaled by safety margin and mild uncertainty
    slack_margin = np.where(slack > 0, slack, 0.0)
    fairness_raw = np.sqrt(np.maximum(ready_wait_time, 0.0) + eps)
    # Only boost fairness when slack is safe *and* ample
    fairness_boost = np.where(slack_margin > 2.0 * margin_clip, 1.0 + 0.3 * np.sqrt(np.clip(uncertainty, 0.0, 1.0)), 1.0)
    fairness_masked = fairness_raw * fairness_boost * (slack_margin > margin_clip).astype(float)
    norm_fairness = normalize_mad(fairness_masked)
    fairness_term = -0.18 * norm_fairness
    
    # Uncertainty penalty: separate, light penalty on high-risk tasks under pressure
    unc_penalty = np.clip(uncertainty, 0.0, 1.0) * dpr
    norm_uncertainty = normalize_mad(unc_penalty)
    uncertainty_term = 0.25 * norm_uncertainty
    
    # Weighted sum with dpr-scaling on urgency dominance
    score = (
        (4.6 + 0.4 * dpr) * norm_urgency +
        ced_term +
        energy_term +
        latency_term +
        fairness_term +
        uncertainty_term +
        hard_ddl_offset
    )
    
    # Final sanitization
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    return score
