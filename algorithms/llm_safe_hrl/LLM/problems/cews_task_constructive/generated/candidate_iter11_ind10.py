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
    v3: Hybrid priority rule combining Parent 2's robust urgency & CED with Parent 1's energy-gating and fairness calibration.
    Key innovations:
    - Adaptive deadline enforcement: triple-mode (arctan/linear/exp) from v2, but scaled by normalized slack margin to avoid over-penalization of large positive slack
    - Critical-energy density (CED) enhanced with Parent 1's *slack-aware gating*: only active when slack >= median_slack (not just >0), preventing premature energy optimization
    - Unified risk penalty: uncertainty * |slack| when slack < 0 (v2) + Parent 1's uncertainty scaling by norm_urgency for multiplicative risk amplification
    - Fairness via clipped tanh wait boost (v1) — more stable than log1p under bursty waits — but gated by slack safety margin and bounded [0,0.15]
    - All normalizations use safe_mad_normalize (v2) for stability, with explicit N=1 handling and outlier clipping
    - Energy term weight increased (2.8) to better reflect objective: DDL-hard constraint first, then minimize energy
    - Final score clamped to finite range and fully deterministic
    """
    eps = 1e-08
    # Clean inputs: replace NaN/inf with safe defaults
    def clean_array(x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    min_exec_time = clean_array(np.asarray(min_exec_time, dtype=float))
    min_comm_time = clean_array(np.asarray(min_comm_time, dtype=float))
    min_incremental_energy = clean_array(np.asarray(min_incremental_energy, dtype=float))
    slack = clean_array(np.asarray(slack, dtype=float))
    upward_rank = clean_array(np.asarray(upward_rank, dtype=float))
    remaining_work = clean_array(np.asarray(remaining_work, dtype=float))
    ready_wait_time = clean_array(np.asarray(ready_wait_time, dtype=float))
    uncertainty = clean_array(np.asarray(uncertainty, dtype=float))
    
    def safe_mad_normalize(x):
        """Robust MAD normalization: handles N=1, constants, outliers; clips to [-4.0, 4.0]"""
        if x.size == 1:
            return np.zeros_like(x)
        x_clipped = np.clip(x, -1000000.0, 1000000.0)
        med = np.median(x_clipped)
        mad = np.median(np.abs(x_clipped - med)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x_clipped - med) / mad
        return np.clip(normed, -4.0, 4.0)
    
    # === Deadline Urgency (Triple-mode, scaled by relative slack margin) ===
    # Compute slack margin: slack relative to its own distribution to avoid bias from absolute scale
    slack_center = np.median(slack) if slack.size > 1 else 0.0
    slack_span = np.percentile(slack, 75) - np.percentile(slack, 25) if slack.size > 1 else 1.0
    slack_span = max(slack_span, eps)
    rel_slack = (slack - slack_center) / (slack_span + eps)  # normalized slack deviation
    
    arctan_urgency = 0.5 + 1.0 / np.pi * np.arctan(np.where(rel_slack >= 0, rel_slack, 0.0))
    linear_violation = np.where((rel_slack < 0) & (rel_slack >= -1.0), 1.0 + -rel_slack, 0.0)
    severe_violation = np.where(rel_slack < -1.0, np.exp(np.clip(-rel_slack - 1.0, 0.0, 20.0)), 0.0)
    deadline_risk_raw = arctan_urgency + linear_violation + severe_violation
    deadline_score = safe_mad_normalize(deadline_risk_raw)
    
    # === Critical-Energy Density (CED) with Slack-Aware Gating (v1 + v2 hybrid) ===
    # Gate energy efficiency term only when slack is safely above median (prevents early energy greed)
    median_slack = np.median(slack) + eps
    ced_gate = (slack >= median_slack).astype(float)
    # CED numerator: importance * work; denominator: energy * latency footprint
    ced_numerator = upward_rank * remaining_work + eps
    ced_denominator = (min_incremental_energy + eps) * (min_exec_time + min_comm_time + eps)
    ced_raw = np.where(ced_gate > 0.0, ced_numerator / ced_denominator, 0.0)
    ced_norm = safe_mad_normalize(ced_raw)
    
    # === Upward Rank (gated by both slack > 0 AND remaining_work > 0, per v2) ===
    upward_rank_active = np.where((slack > 0) & (remaining_work > eps), upward_rank, 0.0)
    upward_rank_norm = safe_mad_normalize(upward_rank_active)
    
    # === Risk Penalty: Uncertainty × |Slack| when violated (v2) + Uncertainty × NormUrgency (v1) ===
    base_risk = np.where(slack < 0, -slack * np.clip(uncertainty, 0.0, 0.7), 0.0)
    # Multiplicative amplification: high uncertainty * high urgency → stronger penalty
    urg_scaled_risk = uncertainty * np.abs(deadline_score)  # use normalized urgency magnitude
    risk_penalty_raw = base_risk + urg_scaled_risk
    risk_penalty_norm = safe_mad_normalize(risk_penalty_raw)
    
    # === Fairness: Clipped tanh wait boost (v1), gated by slack safety margin ===
    median_wait = np.median(ready_wait_time) + eps
    wait_tanh = np.tanh(ready_wait_time / (median_wait + eps))
    # Safety margin: only apply fairness boost when slack is non-negative and above threshold
    slack_safety_ratio = np.clip(slack, 0.0, 1000000.0) / (np.clip(np.mean(np.clip(slack, 0.0, np.inf)) + eps, eps, 1000.0) + eps)
    wait_boost = wait_tanh * np.clip(slack_safety_ratio, 0.0, 1.0)
    wait_boost = np.clip(wait_boost, 0.0, 0.15)  # tighter bound than v1
    fairness_norm = safe_mad_normalize(wait_boost)
    
    # === Weighted combination (strict hierarchy: deadline > energy > critical path > risk > fairness) ===
    w_deadline = 5.2      # slightly increased for harder DDL enforcement
    w_ced = 2.8          # increased vs v2 to better prioritize energy under safety
    w_upward = 1.4       # slightly reduced to de-emphasize pure CP when energy matters
    w_risk = 0.45        # increased vs v2 to strengthen violation coupling
    w_fairness = 0.07    # balanced between v1/v2 stability
    
    score = (
        w_deadline * deadline_score
        - w_ced * ced_norm
        - w_upward * upward_rank_norm
        + w_risk * risk_penalty_norm
        + w_fairness * fairness_norm
    )
    
    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    
    return score
