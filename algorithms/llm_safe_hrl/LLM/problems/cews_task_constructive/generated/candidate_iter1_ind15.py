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
    """Novel priority heuristic emphasizing deadline urgency, critical-path leverage,
    and risk-aware energy efficiency — with starvation prevention and robust scaling.

    Key innovations:
    - Uses *slack-based urgency ratio* (1 / (1 + max(0, slack))) for smooth, bounded
      penalty on tight/negative deadlines — avoids division-by-zero and explodes only
      near critical slack=0, decaying gracefully for large slack.
    - Introduces *energy-per-critical-work* ratio: min_incremental_energy / (remaining_work + eps),
      rewarding tasks that deliver high computation impact per joule — promotes energy-efficient
      progress on important subpaths.
    - Combines upward_rank and remaining_work into a *critical-leverage score*:
      upward_rank * (1 + normalize(remaining_work)), amplifying importance of high-rank,
      high-work tasks without unbounded growth.
    - Applies *waiting-time boost* only after threshold (ready_wait_time > median), preventing
      premature boosting while ensuring fairness for long-waiting tasks.
    - All features normalized via robust *IQR-based scaling* (interquartile range + eps) instead
      of mean-abs — more resistant to outliers in heterogeneous workflows.
    - Final score is convex combination of three orthogonal objectives:
        (1) Deadline urgency (dominant when slack < 0),
        (2) Critical-path energy efficiency,
        (3) Starvation-aware fairness.
    """
    eps = 1e-8

    # Ensure float arrays without modifying inputs
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust IQR-based normalization: scale = x / (IQR(x) + eps)
    def iqr_normalize(x):
        q75, q25 = np.percentile(x, [75, 25], method='midpoint') if x.size > 1 else (x[0], x[0])
        iqr = q75 - q25
        return x / (iqr + eps)

    # === (1) Deadline urgency: smooth, bounded, monotonic penalty for small/negative slack ===
    # For slack <= 0 → urgency → inf; for slack > 0 → urgency decays as 1/(1+slack)
    # Clamped to avoid overflow: max(urgency) = 1e3
    urgency_base = np.where(slack <= 0, 1e3, 1.0 / (1.0 + slack))
    urgency = iqr_normalize(urgency_base)

    # === (2) Critical-path energy efficiency: energy per unit critical work ===
    # Encourages assigning low-energy VMs to high-upward-rank, high-remaining-work tasks
    energy_per_crit_work = min_incremental_energy / (remaining_work + eps)
    # Weight by upward_rank to reflect path importance
    crit_eff_score = upward_rank * (1.0 + iqr_normalize(remaining_work)) * iqr_normalize(energy_per_crit_work)
    # Normalize final composite
    crit_eff = iqr_normalize(crit_eff_score)

    # === (3) Starvation-aware waiting boost: only activate after median wait time ===
    wait_thresh = np.median(ready_wait_time) if ready_wait_time.size > 0 else 0.0
    wait_boost_raw = np.where(ready_wait_time > wait_thresh, ready_wait_time - wait_thresh, 0.0)
    wait_boost = iqr_normalize(wait_boost_raw)

    # === (4) Uncertainty-adjusted communication penalty ===
    # High uncertainty amplifies comm cost impact (since timing is less predictable)
    comm_risk_adjusted = min_comm_time * (1.0 + iqr_normalize(uncertainty))
    comm_penalty = iqr_normalize(comm_risk_adjusted)

    # === Final score: convex combination with deadline urgency dominant ===
    # Weights sum to 1.0; urgency gets highest weight to enforce DDL feasibility first.
    score = (
        0.45 * urgency               # Highest weight: hard DDL enforcement
        + 0.30 * crit_eff            # Medium: energy-efficient critical progress
        + 0.15 * comm_penalty        # Medium: penalize uncertain comms
        + 0.10 * wait_boost          # Low but essential: anti-starvation
    )

    # Numerical safety: clamp and sanitize
    score = np.clip(score, -1e10, 1e10)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    return score
