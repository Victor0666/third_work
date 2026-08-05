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
    critical-path awareness with workload-normalized urgency, and starvation-avoiding
    waiting-time modulation — all under robust per-feature normalization.

    Key innovations:
      - Uses *slack-aware gating*: only tasks with slack < 0 get strong risk penalty;
        others get zero risk term, avoiding over-penalizing safe tasks.
      - Replaces linear HEFT rank weighting with *criticality density*: 
        upward_rank / (remaining_work + eps), rewarding high-rank tasks that carry
        relatively little remaining work (i.e., bottlenecks near end of DAG).
      - Combines execution + communication into *time-pressure* = min_exec_time + min_comm_time,
        then normalizes it jointly to avoid double-counting latency effects.
      - Introduces *energy-efficiency ratio*: min_incremental_energy / (min_exec_time + min_comm_time + eps),
        prioritizing low-energy-per-unit-time tasks when deadlines permit.
      - Applies *waiting-time boost* only after a threshold (e.g., > 1s), preventing noise
        from sub-second jitter while ensuring fairness for truly stalled tasks.
      - Uses *uncertainty-adjusted urgency*: multiplies deadline risk by (1 + uncertainty),
        amplifying penalty for high-risk tasks without blowing up scores.
      - All features normalized by median absolute deviation (MAD) instead of mean abs,
        improving robustness against outliers in heterogeneous workflows.
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

    def robust_normalize(x):
        """Normalize using median absolute deviation for outlier resilience."""
        x_abs = np.abs(x)
        center = np.median(x_abs)
        spread = np.median(np.abs(x_abs - center)) + eps
        return (x - np.median(x)) / spread

    # Time pressure: total minimal latency footprint (exec + comm)
    time_pressure = min_exec_time + min_comm_time

    # Energy efficiency ratio: joules per second of minimal latency; higher is worse
    energy_eff_ratio = min_incremental_energy / (time_pressure + eps)

    # Deadline risk: only active when slack < 0; amplified by uncertainty
    deadline_risk = np.where(slack < 0, -slack * (1.0 + uncertainty), 0.0)

    # Criticality density: upward_rank per unit remaining work — identifies high-impact, lightweight bottlenecks
    criticality_density = upward_rank / (remaining_work + eps)

    # Waiting boost: only activated after 1s, scaled linearly but capped at 5x median wait
    wait_boost = np.where(
        ready_wait_time > 1.0,
        np.clip(ready_wait_time / (np.median(ready_wait_time) + eps), 0.0, 5.0),
        0.0
    )

    # Normalize each component independently (robust MAD-based)
    norm_time_pressure = robust_normalize(time_pressure)
    norm_energy_eff = robust_normalize(energy_eff_ratio)
    norm_deadline_risk = robust_normalize(deadline_risk)
    norm_criticality = robust_normalize(criticality_density)
    norm_wait_boost = robust_normalize(wait_boost)
    norm_uncertainty = robust_normalize(uncertainty)

    # Priority score: smaller = better
    # Strong negative weight on deadline risk ensures urgent tasks dominate.
    # Positive weights on time_pressure & energy_eff push for efficient, low-latency scheduling.
    # Negative weight on criticality_density favors bottleneck resolution.
    # Small positive weight on wait_boost prevents starvation without overwhelming DDL safety.
    score = (
        0.30 * norm_time_pressure
        + 0.25 * norm_energy_eff
        - 3.00 * norm_deadline_risk
        - 0.15 * norm_criticality
        + 0.05 * norm_wait_boost
        + 0.10 * norm_uncertainty
    )

    # Final numerical safeguard
    return np.nan_to_num(
        score,
        nan=1e12,
        posinf=1e12,
        neginf=-1e12
    )
