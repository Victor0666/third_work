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
    Hybrid priority rule: combines Parent 2's deadline-safety and critical-path precision
    with Parent 1's robust normalization, additive risk gating, and starvation control.
    
    Key innovations:
    - Uses slack-margin gating (Parent 2) but adds *adaptive urgency scaling*: linear penalty only for slack < -0.1s,
      then flat saturation beyond -5s to prevent outlier domination.
    - Keeps slack-weighted upward rank (sigmoid(-slack/5)) from Parent 2 for critical-path continuity near deadlines.
    - Replaces IQR normalization with Parent 1's adaptive min-max/std fallback — more stable for small N and degenerate cases.
    - Energy term uses joules-per-critical-time (Parent 2), but normalized *before* combination to avoid scale leakage.
    - Aging is slack-conditioned (Parent 2) but capped at 5% of base score and applied additively (Parent 1 style).
    - Risk gating is *additive and conditional*: only active when (slack < -0.1) AND (uncertainty > median_uncertainty),
      avoiding noise amplification under low uncertainty.
    - All operations are division-safe, NaN/inf-guarded, and guarantee finite (N,) output.
    """
    eps = 1e-08
    # Cast and sanitize inputs
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    # Adaptive normalization: min-max if meaningful range, else std, else 1.0
    def normalize_adaptive(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        rng = np.max(x) - np.min(x)
        if rng > eps and len(np.unique(x)) > 1:
            center = np.mean(x)
            scale = rng
        else:
            std_val = np.std(x)
            center = np.median(x)
            scale = std_val if std_val > eps else 1.0
        return (x - center) / (scale + eps)
    
    # === Deadline urgency ===
    # Slack-margin gating: only penalize when slack < -0.1s; saturate beyond -5s for stability
    deadline_risk = np.clip(np.where(slack < -0.1, -slack, 0.0), 0.0, 5.0)
    urgency = normalize_adaptive(deadline_risk)
    
    # === Critical-path importance ===
    # Sigmoid-weighted upward rank preserves priority for marginally tight tasks (Parent 2)
    slack_sigmoid = 1.0 / (1.0 + np.exp(-slack / 5.0))
    weighted_upward_rank = upward_rank * slack_sigmoid
    norm_weighted_rank = normalize_adaptive(weighted_upward_rank)
    
    # === Energy efficiency ===
    # Joules-per-critical-time (Parent 2), decoupled from work volume
    exec_comm_safe = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_time = min_incremental_energy / exec_comm_safe
    norm_energy = normalize_adaptive(energy_per_time)
    
    # === Aging boost ===
    # Slack-conditioned (tanh) and capped at 5% of base urgency magnitude (Parent 1+2 hybrid)
    aging_denom = np.abs(slack) + 1.0
    aging_boost_raw = np.tanh(0.3 * ready_wait_time / aging_denom)
    # Cap aging contribution to avoid overriding deadline signals
    aging_boost = 0.05 * np.abs(urgency) * aging_boost_raw
    
    # === Risk gating ===
    # Additive, not multiplicative: only activate risk penalty when both deadline risk AND high uncertainty exist
    median_uncertainty = np.median(uncertainty) if uncertainty.size > 0 else 0.0
    risk_active = (slack < -0.1) & (uncertainty > median_uncertainty + eps)
    risk_penalty = np.where(risk_active, uncertainty * 0.25, 0.0)
    norm_risk = normalize_adaptive(risk_penalty)
    
    # === Final score: weighted sum with strict monotonicity in deadline terms ===
    # Prioritize deadline safety first, then critical path, then energy, then controlled aging/risk
    score = (
        0.40 * urgency +           # Primary deadline enforcement
        0.25 * norm_weighted_rank + # Critical path fidelity
        0.20 * norm_energy +        # Energy efficiency (scaled down vs deadline)
        0.10 * norm_risk +          # Conditional risk penalty
        0.05 * aging_boost          # Starvation mitigation (capped, additive)
    )
    
    # Ensure finite, deterministic output with shape (N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
