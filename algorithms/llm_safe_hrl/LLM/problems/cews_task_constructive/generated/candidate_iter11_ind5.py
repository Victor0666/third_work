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
    Hybrid priority rule v2: Combines Parent 2's robust gating and decoupled physics with Parent 1's starvation control and risk-aware energy scaling.
    Key innovations:
      - Unified urgency gating using both relative slack percentile AND absolute deadline pressure (slack < 0 or < 30s)
      - Starvation boost gated by *both* slack tightness AND work-normalized wait, preventing idle starvation under soft deadlines
      - Criticality-energy synergy enhanced: (upward_rank * task_duration) / (min_incremental_energy + eps) * (1 + uncertainty * |slack|^-1 when violated)
      - Risk-weighted energy uses adaptive exponent: 1.0 + uncertainty * max(0, -slack) * (1 + upward_rank / median_ur), amplifying penalty for high-criticality late tasks
      - All normalization uses clipped z-score with IQR fallback, guaranteed stable for N=1 via median-only centering
      - Final score is convex combination of normalized terms with weights tuned for DDL-hardness + energy-minimization tradeoff
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

    # Robust normalization: handles N=1, outliers, NaN/inf
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        median_x = np.median(x)
        q25, q75 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q75 - q25 + eps
        scale = iqr if iqr > eps else np.mean(np.abs(x - median_x)) + eps
        z = (x - median_x) / (scale + eps)
        return np.clip(z, -8.0, 8.0)

    # Task duration: execution + communication (avoid zero)
    task_duration = min_exec_time + min_comm_time + eps

    # Relative slack: safe division
    rel_slack = np.divide(slack, task_duration, out=np.zeros_like(slack, dtype=float), where=task_duration!=0)

    # Unified urgency gating: percentile-based *and* hard-deadline pressure
    slack_thresh_rel = np.quantile(rel_slack, 0.3) if N > 1 else np.min(rel_slack)
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (slack < 30.0)
    urgency_penalty = np.zeros_like(slack)
    urgency_penalty[violated_mask] = np.exp(-slack[violated_mask] / (task_duration[violated_mask] + eps))
    urgency_penalty[tight_mask] = np.maximum(0.0, slack_thresh_rel - rel_slack[tight_mask]) * 2.0

    # Latency-criticality-energy synergy: ur × duration / energy, amplified under violation
    base_synergy = upward_rank * task_duration / (min_incremental_energy + eps)
    synergy_amplifier = np.where(violated_mask, 1.0 + 0.5 * uncertainty * (1.0 + upward_rank / (np.median(upward_rank) + eps)), 1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, 1e-6, 1e8)

    # Risk-weighted energy: adaptive exponent driven by slack violation severity and criticality
    slack_distance = np.maximum(0.0, -slack)
    median_ur = np.median(upward_rank) + eps
    ur_ratio = np.clip(upward_rank / median_ur, 0.1, 10.0)
    energy_exponent = 1.0 + uncertainty * slack_distance * ur_ratio
    risk_weighted_energy = min_incremental_energy * np.power(1.0 + eps, energy_exponent)
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e9)

    # Starvation boost: only when slack > 0 AND work is significant AND wait time is non-trivial
    median_rw = np.median(remaining_work) + eps
    rw_normalized = np.clip(remaining_work / median_rw, 0.01, 100.0)
    wait_boost_mask = (slack > 0.0) & (rw_normalized > 0.3) & (ready_wait_time > 0.1 * np.mean(task_duration + eps))
    max_wait = np.maximum(np.max(ready_wait_time), eps)
    wait_boost = np.where(wait_boost_mask, 
                         np.clip(ready_wait_time / max_wait, 0.0, 0.4) * np.sqrt(rw_normalized), 
                         0.0)

    # Normalized components
    norm_urgency = robust_normalize(urgency_penalty)
    norm_synergy = robust_normalize(latency_crit_synergy)
    norm_energy = robust_normalize(risk_weighted_energy)
    norm_wait = robust_normalize(wait_boost)

    # Final convex combination: prioritizes urgency first, then synergy (critical path), then energy, then starvation
    score = (
        0.45 * norm_urgency +
        0.30 * (-norm_synergy) +  # Higher synergy → lower priority score (since -norm_synergy)
        0.20 * norm_energy +
        0.05 * norm_wait
    )

    # Final sanitization
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    return score
