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
    Mutated priority rule emphasizing:
    - Slack-driven urgency with *adaptive penalty curvature* (quadratic near zero, linear beyond)
    - Energy-efficiency via *risk-adjusted energy-per-duration*, robustly capped and normalized
    - Criticality-aware fairness: upward_rank modulates starvation penalty only when slack is tight
    - Uncertainty gating: uncertainty boosts priority *only* when slack <= 0 and uncertainty > median
    - Work-normalized energy: scale min_incremental_energy by sqrt(remaining_work) for downstream impact awareness
    - All components IQR-normalized with sign-preserving centering; final score bounded and deterministic.
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
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_normalize(x):
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1 + eps
        center = np.median(x)
        normed = (x - center) / iqr
        return np.clip(normed, -10.0, 10.0)

    # --- 1. Adaptive deadline urgency: quadratic near slack=0, linear beyond ---
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = slack / task_duration
    # Quadratic penalty for small negative slack; linear for large negative slack
    urgency_base = np.where(
        rel_slack <= 0.0,
        np.where(rel_slack >= -1.0, 1.0 + (rel_slack)**2 * 2.0, 1.0 + (-rel_slack) * 1.5),
        np.exp(-rel_slack * 0.3)  # Soft decay for positive slack
    )

    # --- 2. Risk-adjusted energy efficiency: energy per effective duration ---
    # Effective duration includes uncertainty margin only under risk (slack <= 0)
    risk_duration = np.where(
        slack <= 0.0,
        task_duration * (1.0 + np.clip(uncertainty, 0.0, 2.0)),
        task_duration
    )
    energy_per_sec = np.maximum(min_incremental_energy, eps) / np.maximum(risk_duration, eps)
    # Cap extreme ratios to avoid outlier dominance
    energy_per_sec = np.clip(energy_per_sec, 1e-6, 1e6)
    energy_eff_norm = robust_normalize(energy_per_sec)

    # --- 3. Criticality-weighted starvation control ---
    # Only apply waiting penalty if task is both high-criticality AND slack-constrained
    median_ur = np.median(upward_rank) + eps
    is_critical = (upward_rank > median_ur).astype(float)
    is_urgent = (slack <= 0.0).astype(float)
    # Scale wait penalty by criticality & urgency; bound total contribution to [0, 0.3]
    max_wait = np.maximum(np.max(ready_wait_time), eps)
    raw_wait = np.clip(ready_wait_time / max_wait, 0.0, 1.0)
    wait_penalty = raw_wait * is_critical * is_urgent * 0.3

    # --- 4. Uncertainty gating: boost priority only under joint risk (slacking + high uncertainty)
    median_unc = np.median(uncertainty) + eps
    unc_boost = np.where(
        (slack <= 0.0) & (uncertainty > median_unc),
        np.clip(uncertainty / median_unc, 1.0, 3.0),
        1.0
    )
    unc_norm = robust_normalize(uncertainty)

    # --- 5. Work-normalized energy: account for descendant impact ---
    # Scale energy by sqrt(remaining_work) to reflect aggregate downstream cost
    work_scale = np.sqrt(np.maximum(remaining_work, eps))
    work_scaled_energy = min_incremental_energy * work_scale
    work_energy_norm = robust_normalize(work_scaled_energy)

    # --- 6. Upward rank normalized and gated by slack ---
    ur_norm = robust_normalize(upward_rank)
    # Reduce criticality weight when slack is ample
    ur_weight = np.where(slack > 0.0, 0.5, 1.0)
    ur_gated = ur_norm * ur_weight

    # --- Combine all components with interpretable weights ---
    # Higher urgency → lower score (so we negate urgency_base)
    # Lower energy_per_sec → lower score → keep as-is
    # All normalized terms are centered at 0; smaller final score = higher priority
    score = (
        -2.8 * urgency_base      # Dominant deadline driver
        + 1.4 * energy_eff_norm  # Energy efficiency: smaller = better
        + 0.4 * ur_gated         # Critical path support, gated
        + 0.25 * work_energy_norm  # Downstream energy impact
        + 0.2 * unc_norm         # Uncertainty signal, normalized
        + wait_penalty           # Bounded starvation control
    )

    # Final safeguard: replace NaN/inf and clamp extremes
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e10, 1e10)

    return score
