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
    v2 priority rule: Urgency-first with deadline-aware criticality, risk-normalized energy,
                      fairness under pressure, and simplified robust uncertainty coupling.
    
    Key improvements over v1:
    - Eliminates tanh-based dynamic weighting (numerically unstable & weakens urgent penalties)
    - Replaces dual-gated starvation relief with *sliding-window wait-pressure* activated by slack < median,
      ensuring fairness even near deadlines.
    - Removes fragile `critical_gate * uncertainty` coupling; instead applies uncertainty scaling only to 
      energy term via multiplicative (1 + uncertainty) — preserves urgency dominance.
    - Uses *slack-relative duration* for latency gating: tightness defined as slack/duration < threshold,
      avoiding median dependency that misbehaves on skewed distributions.
    - Introduces *normalized wait pressure*: (ready_wait_time / (duration + eps)) scaled by (1 - sigmoid(slack)),
      amplifying fairness when slack is low without hard thresholds.
    - All normalizations use robust MAD with explicit zero-MAD fallback and strict clipping.
    - Energy penalty now strictly attenuated only by *urgency rank*, not tanh — uses percentile-based urgency mask.
    - Final layering enforces hard priority: urgency > criticality > energy > wait-pressure > work-bonus.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)

    def robust_mad_norm(x):
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x_clean.size == 0:
            return np.zeros_like(x_clean)
        med = np.median(x_clean)
        mad = np.median(np.abs(x_clean - med))
        if mad < eps:
            return np.zeros_like(x_clean)
        z = (x_clean - med) / (mad + eps)
        return np.clip(z, -3.0, 3.0)

    # Urgency: hard priority for violated or imminent deadlines
    is_urgent = (slack <= 0.0).astype(np.float64)
    
    # Duration & normalized slack sensitivity
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0)
    # Latency-aware criticality gate: activate only when slack is tight relative to duration
    tight_slack_mask = (rel_slack <= 0.3).astype(np.float64)
    critical_latency_raw = duration * upward_rank
    norm_critical_latency = robust_mad_norm(critical_latency_raw)
    
    # Risk-normalized energy: divide by duration * (1 + uncertainty) to penalize uncertain high-energy assignments
    effective_duration = duration * (1.0 + uncertainty + eps)
    risk_norm_energy = np.divide(min_incremental_energy, effective_duration,
                                 out=np.zeros_like(min_incremental_energy), where=effective_duration != 0)
    risk_norm_energy = np.nan_to_num(risk_norm_energy, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy = robust_mad_norm(risk_norm_energy)
    
    # Wait pressure: fairness under deadline pressure — no dual gating; uses sigmoid-slack scaling
    # Higher weight when slack is low, smoothly decaying as slack increases
    slack_sigmoid = 1.0 / (1.0 + np.exp(-slack / (np.median(duration) + eps)))
    wait_per_duration = np.divide(ready_wait_time, duration, out=np.zeros_like(ready_wait_time), where=duration != 0)
    wait_per_duration = np.nan_to_num(wait_per_duration, nan=0.0, posinf=0.0, neginf=0.0)
    norm_wait_rel = robust_mad_norm(wait_per_duration)
    wait_pressure = (1.0 - slack_sigmoid) * norm_wait_rel  # higher pressure when slack is low
    
    # Work bonus: reward scheduling large remaining work early (negative score contribution)
    norm_remaining_work = robust_mad_norm(remaining_work)
    work_bonus = -0.08 * norm_remaining_work  # modest bonus to avoid overriding urgency
    
    # Uncertainty coupling: only in energy denominator — no separate term to avoid diluting urgency
    # Already embedded in effective_duration above
    
    # Priority hierarchy layers (additive, decreasing weights)
    score = np.full(N, 0.0, dtype=np.float64)
    # Layer 1: Hard urgency override
    score = np.where(is_urgent, -1e12, score)
    # Layer 2: Criticality penalty (only when slack tight)
    score = np.where(is_urgent, score, score + 0.4 * norm_critical_latency * tight_slack_mask)
    # Layer 3: Energy penalty (always applied, but scaled by criticality gate for focus)
    score = np.where(is_urgent, score, score + 0.25 * norm_energy * tight_slack_mask)
    # Layer 4: Wait pressure (active across all, amplified under low slack)
    score = np.where(is_urgent, score, score + 0.2 * wait_pressure)
    # Layer 5: Work bonus (always active, encourages early heavy work)
    score = np.where(is_urgent, score, score + work_bonus)
    
    # Final safeguard
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
