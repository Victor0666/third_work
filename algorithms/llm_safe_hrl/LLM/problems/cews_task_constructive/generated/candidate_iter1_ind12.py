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
    """Novel priority rule emphasizing deadline safety via risk-graded slack penalty,
    energy-aware criticality, and starvation-aware waiting boost.
    
    Key innovations:
    - Slack is transformed via smooth, bounded, invertible 'risk sigmoid' that sharply
      penalizes negative slack while preserving ordinality and avoiding division-by-zero.
    - Energy and time features are fused into a single 'energy-delay efficiency' ratio,
      normalized robustly to avoid dominance by outliers.
    - Upward rank is gated by slack: only tasks with sufficient slack (>= 0) receive
      critical-path priority; others yield to deadline urgency.
    - Ready wait time uses square-root scaling for gentle anti-starvation — avoids
      overwhelming deadline signals at large values.
    - Uncertainty is used not as additive cost, but as multiplicative safety margin
      on the effective deadline risk term.
    - All normalizations use median absolute deviation (MAD) for robustness to outliers.
    """
    eps = 1e-8

    # Ensure float64 and copy to avoid mutation
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust normalization: MAD-based (more outlier-resistant than mean-abs)
    def robust_normalize(x):
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        return (x - center) / mad

    # === 1. Deadline Risk Signal: Smooth, bounded, uncertainty-weighted ===
    # Use arctan-based risk mapping: maps (-∞, ∞) → (-π/2, π/2), then shift & scale
    # So negative slack gives strong negative score (→ high priority), zero is neutral,
    # positive slack gives mild positive penalty (to avoid indefinite deferral).
    # Multiply by (1 + uncertainty) to tighten margin for risky tasks.
    risk_factor = 1.0 + np.clip(uncertainty, 0.0, 10.0)  # cap uncertainty
    raw_risk = np.arctan(slack * 0.1)  # scaled to keep sensitivity near zero
    # Invert: we want *smaller* score for *higher* risk → flip sign
    deadline_score = -raw_risk * risk_factor

    # === 2. Energy-Delay Efficiency: Combine exec+comm time into delay, then
    # form energy-per-delay ratio. Low ratio = efficient; we want to prioritize
    # *efficient* tasks *only when safe*, so subtract it (lower ratio → lower score)
    total_delay = np.maximum(min_exec_time + min_comm_time, eps)
    energy_efficiency = min_incremental_energy / total_delay  # J/s = W, but relative
    # Normalize efficiency (lower is better); subtract so low efficiency → higher score
    efficiency_score = robust_normalize(energy_efficiency)

    # === 3. Criticality Gating: Only reward upward_rank if slack >= 0.
    # If slack < 0, set rank contribution to zero — deadline trumps criticality.
    rank_mask = (slack >= 0.0).astype(float)
    rank_score = -robust_normalize(upward_rank) * rank_mask  # negative → prioritize

    # === 4. Anti-Starvation: sqrt-scaled wait time — gentle boost, no explosion
    wait_boost = np.sqrt(np.maximum(ready_wait_time, 0.0) + eps)
    wait_score = -robust_normalize(wait_boost)  # negative → longer wait → higher priority

    # === 5. Workload signal: prefer scheduling high remaining_work *only when slack allows*
    # Prevents fragmentation of heavy sub-DAGs; again gated by slack.
    work_mask = (slack >= -1.0).astype(float)  # allow slight leeway
    work_score = -robust_normalize(remaining_work) * work_mask

    # === 6. Uncertainty as standalone stability penalty (not risk-multiplier here)
    # High uncertainty → mildly penalize (push later unless urgent)
    unc_score = robust_normalize(uncertainty)

    # === Weighted ensemble — all terms aligned: smaller = better
    score = (
        0.30 * deadline_score       # dominant: deadline safety first
        + 0.25 * efficiency_score   # strong: energy efficiency when safe
        + 0.20 * rank_score         # moderate: critical path only if feasible
        + 0.10 * wait_score         # mild: prevent starvation
        + 0.10 * work_score         # mild: preserve heavy-subgraph locality
        + 0.05 * unc_score          # light: discourage uncertain tasks unless urgent
    )

    # Final numerical guard: ensure finite, deterministic output
    return np.nan_to_num(
        score,
        nan=1e12,
        posinf=1e12,
        neginf=-1e12
    )
