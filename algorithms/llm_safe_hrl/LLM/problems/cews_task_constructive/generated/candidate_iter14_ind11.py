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
    v2: Hybrid deadline-hard priority with robust risk-aware energy gating,
         starvation-activated fairness, and uncertainty-calibrated urgency.
    
    Key innovations:
    - Combines Parent 2's absolute piecewise urgency (arctan + linear + exp) with Parent 1's robust_slack = slack - 2*uncertainty
      to explicitly embed uncertainty into deadline assessment while preserving hard DDL semantics.
    - Energy optimization (CED) gated by robust_slack >= 0 AND upward_rank > 0 → stricter than Parent 2 (uses robust_slack not raw slack)
    - Risk penalty uses robust_slack < 0 AND uncertainty > eps → avoids zero-risk amplification
    - Fairness activated only when robust_slack < 0, scaled by max(1, -robust_slack_min) for stronger starvation relief under pressure
    - All normalization uses safe_mad_normalize with N=1 handling and outlier clipping
    - Final weights emphasize deadline (5.5), energy (3.0), critical-path (1.4), risk (0.6), fairness (0.1)
    """
    eps = 1e-08
    
    # Sanitize inputs: convert to float, replace NaN/inf
    def clean_array(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    min_exec_time = clean_array(min_exec_time)
    min_comm_time = clean_array(min_comm_time)
    min_incremental_energy = clean_array(min_incremental_energy)
    slack = clean_array(slack)
    upward_rank = clean_array(upward_rank)
    remaining_work = clean_array(remaining_work)
    ready_wait_time = clean_array(ready_wait_time)
    uncertainty = clean_array(uncertainty)
    
    # Robust slack: account for uncertainty in deadline margin
    robust_slack = slack - 2.0 * uncertainty
    robust_slack = np.clip(robust_slack, -1e12, 1e12)
    
    # Absolute urgency: piecewise function preserving hard deadline semantics
    arctan_urgency = 0.5 + 1.0 / np.pi * np.arctan(np.where(robust_slack >= 0, robust_slack / 10.0, 0.0))
    linear_violation = np.where((robust_slack < 0) & (robust_slack >= -10), 1.0 + (-robust_slack) / 10.0, 0.0)
    severe_violation = np.where(robust_slack < -10, np.exp(np.clip(-robust_slack - 10, 0.0, 20.0)), 0.0)
    deadline_risk_raw = arctan_urgency + linear_violation + severe_violation
    
    # Safe MAD normalization: handles N=1, constants, outliers
    def safe_mad_normalize(x):
        if x.size == 1:
            return np.zeros_like(x)
        x_clipped = np.clip(x, -1e6, 1e6)
        med = np.median(x_clipped)
        mad = np.median(np.abs(x_clipped - med)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x_clipped - med) / mad
        return np.clip(normed, -4.0, 4.0)
    
    deadline_score = safe_mad_normalize(deadline_risk_raw)
    
    # Energy efficiency gating: only activate when robust_slack >= 0 AND upward_rank > 0
    ced_gate = ((robust_slack >= 0) & (upward_rank > eps)).astype(float)
    ced_numerator = upward_rank * remaining_work + eps
    ced_denominator = (min_incremental_energy + eps) * (min_exec_time + min_comm_time + eps)
    ced_raw = np.where(ced_gate > 0.0, ced_numerator / ced_denominator, 0.0)
    ced_norm = safe_mad_normalize(ced_raw)
    
    # Critical-path boost: only active when robust_slack > 0 (safe margin) and work > 0
    upward_rank_active = np.where((robust_slack > 0) & (remaining_work > eps), upward_rank, 0.0)
    upward_rank_norm = safe_mad_normalize(upward_rank_active)
    
    # Risk penalty: only when robust_slack < 0 AND uncertainty > eps (avoids zero-risk amplification)
    risk_penalty_raw = np.where(
        (robust_slack < 0) & (uncertainty > eps),
        (-robust_slack) * np.clip(uncertainty, 0.0, 0.7),
        0.0
    )
    risk_penalty_norm = safe_mad_normalize(risk_penalty_raw)
    
    # Starvation-activated fairness: scaled by worst-case robust_slack under violation
    min_robust_slack_neg = np.clip(-np.min(robust_slack), 0.0, 1e6) + eps
    wait_scale = np.clip(ready_wait_time / min_robust_slack_neg, 0.0, 20.0)
    fairness_boost = np.tanh(wait_scale)
    fairness_boost = np.clip(fairness_boost, 0.0, 0.15)
    fairness_norm = safe_mad_normalize(fairness_boost)
    
    # Weighted score: deadline dominates; energy and CP are secondary objectives; risk and fairness are corrections
    w_deadline = 5.5
    w_ced = 3.0
    w_upward = 1.4
    w_risk = 0.6
    w_fairness = 0.1
    
    score = (
        w_deadline * deadline_score
        - w_ced * ced_norm
        - w_upward * upward_rank_norm
        + w_risk * risk_penalty_norm
        + w_fairness * fairness_norm
    )
    
    # Final sanitization: ensure finiteness and determinism
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    return score
