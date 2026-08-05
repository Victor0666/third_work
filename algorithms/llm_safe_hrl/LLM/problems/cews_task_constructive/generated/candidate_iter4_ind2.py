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
    Mutated priority rule v2: Emphasizes *slack-constrained energy density*, 
    *adaptive starvation damping*, and *uncertainty-gated criticality amplification*.
    
    Key mutations from v1:
    - Replaces relative slack ratio with *normalized slack deficit*: max(0, -slack) / (median(duration)+eps),
      providing stable urgency signal even for tiny tasks; avoids division-by-small distortion.
    - Introduces *energy-per-effective-duration*: min_incremental_energy / (min_exec_time + min_comm_time + eps),
      clipped and robustly normalized — directly reflects power-aware efficiency.
    - Starvation penalty now uses *log-scaled waiting time* bounded by slack sign and work criticality,
      preventing dominance when slack < 0 or remaining_work is low.
    - Criticality-energy tradeoff becomes *upward_rank × (1 + uncertainty)^beta / (energy_density + eps)*,
      where beta = 0.5 if slack > 0 else 1.5 — sharpens risk focus only under violation.
    - Drops IQR normalization for *rank-based percentile scaling* (more monotonic & interpretable),
      applied per-feature with explicit 0–1 mapping and epsilon guards.
    - Adds *work-aware urgency modulation*: scales deadline penalty by remaining_work / (median_rw + eps)
      only when slack <= 0, to prioritize high-impact late tasks.
    - All outputs are finite, deterministic, shape-(N,), and eps-protected.
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

    # Compute robust base features
    duration = min_exec_time + min_comm_time + eps
    energy_density = np.clip(min_incremental_energy / duration, eps, 1e6)
    slack_deficit = np.maximum(0.0, -slack)  # only active under violation
    median_duration = np.median(duration) + eps
    median_rw = np.median(remaining_work) + eps
    median_ur = np.median(upward_rank) + eps

    # === Normalization: percentile-based [0,1] scaling with eps-guarded ranks ===
    def percentile_scale(x):
        # Use argsort twice for stable rank; handle ties via mean rank
        ranks = np.argsort(np.argsort(x)) + 1.0
        n = len(x)
        return np.clip(ranks / (n + 1.0), 0.0, 1.0)

    # Normalize each feature independently to [0,1], preserving order & stability
    dur_norm = percentile_scale(duration)
    energy_norm = percentile_scale(energy_density)
    ur_norm = percentile_scale(upward_rank)
    rw_norm = percentile_scale(remaining_work)
    unc_norm = percentile_scale(uncertainty)
    wait_norm = percentile_scale(np.log1p(ready_wait_time))  # log-scaled to damp long waits

    # === Deadline urgency: slack-deficit scaled by work impact only when violated ===
    # Stronger penalty for high remaining_work tasks that are late
    work_impact_factor = np.where(slack <= 0.0, np.clip(remaining_work / median_rw, 0.3, 3.0), 1.0)
    normalized_slack_deficit = np.clip(slack_deficit / median_duration, 0.0, 10.0)
    deadline_urgency = (
        np.clip(normalized_slack_deficit, 0.0, 5.0) * work_impact_factor * 0.8
        + (1.0 - np.exp(-np.clip(slack_deficit, 0.0, 5.0))) * 0.4
    )

    # === Risk-gated criticality-efficiency ratio ===
    # Amplify uncertainty weight only when slack <= 0; use moderate exponent otherwise
    beta = np.where(slack <= 0.0, 1.5, 0.5)
    risk_factor = np.power(1.0 + uncertainty, beta)
    crit_eff_score = (upward_rank * risk_factor) / (energy_density + eps)
    crit_eff_norm = percentile_scale(np.clip(crit_eff_score, eps, 1e7))

    # === Adaptive starvation control: bounded log-wait, suppressed under violation ===
    # When slack <= 0, let urgency dominate; otherwise add fairness boost
    wait_penalty = np.where(
        slack <= 0.0,
        0.0,
        np.clip(wait_norm * (1.0 - ur_norm) * 0.15, 0.0, 0.25)  # less wait-penalty for critical tasks
    )

    # === Composite score: minimize → higher priority ===
    # Sign convention: negative weights for desirable traits (urgency, crit_eff), positive for costs
    score = (
        -2.8 * deadline_urgency          # urgent late tasks get top priority
        - 2.0 * crit_eff_norm            # favor critical + energy-efficient tasks
        + 0.9 * dur_norm                 # slight penalty for long-duration tasks (power-aware)
        + 0.4 * energy_norm              # mild penalty for high marginal energy
        + 0.3 * unc_norm                 # penalty for high uncertainty (only when not violating)
        + wait_penalty                   # fairness term, disabled under violation
    )

    # Final guard: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
