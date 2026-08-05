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
    Hybrid priority rule v2: Combines Parent 2's robust urgency and slack-aware criticality
    with Parent 1's MAD-based outlier suppression, communication pressure, and starvation control.
    Key innovations:
      - Uses MAD normalization (Parent 1) for tighter outlier handling, with N=1 fallback
      - Integrates work-normalized communication pressure (Parent 1) but gated by slack violation
      - Replaces power-law energy scaling with linear uncertainty amplification (Parent 1) + relative slack distance (Parent 2)
      - Unifies urgency penalty: piecewise-linear from Parent 2 for monotonicity + MAD-normalized slack ratio from Parent 1
      - Starvation boost uses relative wait ratio *and* urgency coupling (Parent 2) but bounded via MAD normalization (Parent 1)
      - Criticality-energy synergy scaled by slack factor *and* uncertainty-gated amplification (hybrid)
      - All operations protected against zero/Nan/inf; deterministic and N=1 safe
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

    # Robust MAD-based normalization (Parent 1 style, N=1 safe)
    def robust_normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        normed = (x - center) / mad
        return np.clip(normed, -10.0, 10.0)

    # Task duration and derived metrics
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = np.divide(slack, task_duration, out=np.zeros_like(slack), where=task_duration != 0)
    
    # Piecewise-linear urgency penalty (Parent 2) — monotonic, bounded
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (slack < 30.0)
    urgency_penalty = np.zeros_like(slack)
    urgency_penalty[violated_mask] = np.clip(-slack[violated_mask] / (task_duration[violated_mask] + eps), 0.0, 5.0)
    # For near-deadline: use quantile threshold to avoid over-penalizing short tasks
    slack_thresh_rel = np.quantile(rel_slack, 0.3) if N > 1 else np.min(rel_slack)
    urgency_penalty[tight_mask] = np.clip(slack_thresh_rel - rel_slack[tight_mask], 0.0, 3.0)

    # Slack-aware upward rank dampening (Parent 2): reduce criticality weight when slack is ample
    slack_factor = np.clip(1.0 - np.maximum(0.0, slack) / (task_duration + eps), 0.1, 1.0)
    dampened_ur = upward_rank * slack_factor

    # Criticality-energy synergy: latency-critical work per energy unit, amplified under violation
    base_synergy = dampened_ur * task_duration / (min_incremental_energy + eps)
    # Uncertainty-gated amplification only during violation (Parent 2)
    median_ur = np.median(upward_rank) + eps
    ur_ratio = np.clip(upward_rank / median_ur, 0.1, 10.0)
    synergy_amplifier = np.where(violated_mask, 1.0 + 0.6 * uncertainty * ur_ratio, 1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, 1e-08, 1e8)

    # Risk-weighted energy: linear uncertainty amplification (Parent 1) + relative slack distance (Parent 2)
    rel_slack_distance = np.maximum(0.0, -slack) / (task_duration + eps)
    risk_weighted_energy = min_incremental_energy * (1.0 + uncertainty * rel_slack_distance)
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e9)

    # Communication pressure: high comm/low work prioritized under tight slack (Parent 1), gated by violation
    comm_to_work_ratio = min_comm_time / (remaining_work + eps)
    comm_pressure = np.where(violated_mask, comm_to_work_ratio * 1.8, 
                            np.where(tight_mask, comm_to_work_ratio * 0.4, comm_to_work_ratio * 0.1))

    # Starvation boost: normalized wait ratio coupled with remaining work and slack (Parent 2), bounded
    median_task_dur = np.median(task_duration) + eps
    norm_wait_ratio = np.clip(ready_wait_time / median_task_dur, 0.0, 5.0)
    median_rw = np.median(remaining_work) + eps
    rw_normalized = np.clip(remaining_work / median_rw, 0.01, 100.0)
    wait_boost_mask = (norm_wait_ratio > 0.5) & (rw_normalized > 0.4) & (slack < 30.0)
    wait_boost = np.where(wait_boost_mask, 
                         np.clip(norm_wait_ratio * np.maximum(0.0, 30.0 - slack), 0.0, 1.5), 
                         0.0)

    # Normalize all components using MAD
    norm_urgency = robust_normalize_mad(urgency_penalty)
    norm_synergy = robust_normalize_mad(latency_crit_synergy)
    norm_energy = robust_normalize_mad(risk_weighted_energy)
    norm_comm = robust_normalize_mad(comm_pressure)
    norm_wait = robust_normalize_mad(wait_boost)

    # Final convex combination: prioritize urgency first (0.5), then criticality-energy synergy (-0.35),
    # communication pressure (0.1), energy (0.05), starvation (0.05) — balanced & deadline-hard
    score = (
        0.50 * norm_urgency +
        -0.35 * norm_synergy +
        0.10 * norm_comm +
        0.05 * norm_energy +
        0.05 * norm_wait
    )

    # Final cleanup: ensure finite values and correct shape
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    return score
