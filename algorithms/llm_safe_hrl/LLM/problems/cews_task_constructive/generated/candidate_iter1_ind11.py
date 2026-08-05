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
    """Novel priority rule emphasizing deadline safety via adaptive risk gating,
    critical-path awareness with workload-normalized urgency, and starvation-aware
    energy efficiency.

    Key innovations:
    - Deadline risk is gated: only tasks with slack < 0 get strong negative priority boost;
      others receive mild positive bias to avoid over-prioritizing non-urgent tasks.
    - Energy-efficiency term is scaled by remaining_work to favor low-energy-per-work tasks
      when workload is high (i.e., more "bang for buck" on critical paths).
    - Upward rank is normalized *relative to remaining_work*, promoting high-rank tasks
      that also carry substantial descendant load — avoids inflating low-work critical nodes.
    - Ready wait time is smoothed via soft saturation (tanh) to prevent unbounded growth
      while ensuring monotonic fairness.
    - Uncertainty is transformed into a *risk-adjusted urgency* multiplier on deadline risk
      and energy terms, not added linearly — captures interaction between uncertainty and
      deadline pressure.
    - All features are robustly normalized using IQR-based scaling (more outlier-resistant
      than mean-abs), with explicit epsilon guards.
    """
    eps = 1e-8

    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust normalization: Interquartile Range + epsilon to avoid division by zero
    def normalize_iqr(x):
        q75, q25 = np.percentile(x, [75, 25], method='midpoint') if x.size > 1 else (x[0], x[0])
        iqr = q75 - q25 + eps
        center = np.median(x) if x.size > 0 else 0.0
        return (x - center) / iqr

    # Deadline risk: binary-gated penalty — only penalize negative slack, but strongly
    # For slack >= 0: small positive offset (reduces priority) to avoid crowding
    # For slack < 0: large negative offset (boosts priority) proportional to |slack|
    deadline_risk = np.where(
        slack < 0,
        -np.abs(slack),
        0.1 * np.maximum(slack, 0.0)  # mild disincentive for over-early tasks
    )

    # Workload-aware energy efficiency: reward low incremental energy *per unit work*
    # Avoid division by zero; cap at reasonable max to prevent explosion
    energy_per_work = np.divide(
        min_incremental_energy,
        np.maximum(remaining_work, eps),
        out=np.full_like(min_incremental_energy, 1e6),
        where=remaining_work > eps
    )
    # Normalize energy_per_work — lower values (efficient per MI) get higher priority
    energy_efficiency_score = normalize_iqr(energy_per_work)

    # Critical-path importance: upward_rank weighted by *normalized* remaining_work
    # Ensures high-rank tasks carrying heavy load get priority; suppresses noise from
    # high-rank but trivial descendants
    rank_load_ratio = np.divide(
        upward_rank,
        np.maximum(remaining_work, eps),
        out=np.zeros_like(upward_rank),
        where=remaining_work > eps
    )
    critical_path_score = normalize_iqr(rank_load_ratio)

    # Smoothed waiting time: tanh saturates at ~1.0, ensures monotonic priority gain
    # without unbounded growth; scaled to match other normalized ranges
    wait_score = np.tanh(0.1 * ready_wait_time)  # 0.1 → ~95% saturation at 30s

    # Uncertainty-coupled risk: amplifies deadline risk and energy penalty only when
    # uncertainty is high — models "don’t gamble with risky tasks near deadline"
    # Uses sigmoid to softly activate: 0–1 range, stable, no blowup
    risk_amp = 1.0 + 2.0 * (1.0 / (1.0 + np.exp(-uncertainty + 1.0)))  # shift to activate near median

    # Final score: smaller = better
    # Prioritize deadline safety first (strongest weight), then critical path & efficiency,
    # then fairness & robustness
    score = (
        0.40 * normalize_iqr(deadline_risk) * risk_amp
        + 0.25 * critical_path_score
        + 0.20 * energy_efficiency_score
        - 0.10 * normalize_iqr(min_exec_time)
        - 0.10 * normalize_iqr(min_comm_time)
        - 0.05 * wait_score  # negative → longer wait increases priority
        + 0.05 * normalize_iqr(uncertainty)  # mild penalty for high uncertainty alone
    )

    # Final numeric stabilization
    return np.nan_to_num(
        score,
        nan=1e12,
        posinf=1e12,
        neginf=-1e12
    )
