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
    Self-evolved priority rule v2: Fixes instability from ratio-based starvation and rank-normalization,
    restores robust scale-invariant normalization, and refines risk gating to operate on raw penalties.
    Key improvements:
      - Replaces quantile ranking with clipped z-score normalization (robust, scale-preserving, handles N=1)
      - Starvation index simplified to linear wait boost gated *only* by negative/low slack (no division-by-zero risk)
      - Risk factor applied *before* normalization to preserve physical meaning and avoid rank distortion
      - Slack penalty now includes workload scaling *and* uncertainty amplification only when slack < 0 or < 30s
      - Critical energy term uses upward_rank + eps denominator and is multiplied by (1 + uncertainty) *only under deadline pressure*
      - Latency coupling retained but simplified: min_comm_time * (1 + upward_rank / (median_ur + eps))
      - All intermediate arrays explicitly guarded against NaN/inf at creation; final score bounded deterministically
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

    # Robust normalization: quantile-based IQR scaling with degenerate fallback for small N
    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.array([0.0])
        q25, q75 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q75 - q25
        center = np.median(x)
        scale = iqr if iqr > eps else np.mean(np.abs(x - center)) + eps
        z = (x - center) / (scale + eps)
        return np.clip(z, -10.0, 10.0)

    # Slack penalty: hard exponential for violations, sigmoid for tight slack (<30s), scaled by work & uncertainty
    slack_penalty = np.zeros_like(slack)
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (slack < 30.0)
    slack_penalty[violated_mask] = np.exp(-slack[violated_mask])
    slack_penalty[tight_mask] = 1.0 / (1.0 + np.exp((slack[tight_mask] - 15.0) / 3.0))
    
    # Work scaling: amplify penalty for high remaining_work tasks near deadline
    work_scale = np.clip(remaining_work / (np.median(remaining_work) + eps), 0.5, 5.0)
    slack_penalty = slack_penalty * work_scale

    # Uncertainty amplification only in high-risk regime (slack < 30s OR violated)
    risk_regime = (slack < 30.0) | violated_mask
    slack_penalty = np.where(risk_regime, slack_penalty * (1.0 + 0.5 * uncertainty), slack_penalty)

    # Critical energy: marginal energy per unit criticality, risk-amplified only under deadline pressure
    critical_energy_base = min_incremental_energy / (upward_rank + eps)
    critical_energy_score = np.where(risk_regime, critical_energy_base * (1.0 + 0.5 * uncertainty), critical_energy_base)

    # Latency coupling: communication time weighted by relative criticality
    median_ur = np.median(upward_rank) + eps
    comm_latency_weighted = min_comm_time * (1.0 + upward_rank / (median_ur + eps))

    # Starvation index: linear wait boost, activated only when slack <= 60s (no division, no NaN)
    starvation_boost = np.where(slack <= 60.0, ready_wait_time * (1.0 - np.clip(slack / 60.0, 0.0, 1.0)), 0.0)

    # Normalize each component *after* risk gating — preserves scale and avoids rank distortion
    norm_slack = robust_normalize(slack_penalty)
    norm_energy = robust_normalize(critical_energy_score)
    norm_comm = robust_normalize(comm_latency_weighted + eps)
    norm_starv = robust_normalize(starvation_boost)

    # Final weighted score: urgency dominates, then risk-energy, latency, fairness
    score = (
        0.55 * norm_slack +
        0.30 * norm_energy +
        0.10 * norm_comm +
        0.05 * norm_starv
    )

    # Final sanitization: ensure finite, bounded output
    score = np.nan_to_num(score, nan=0.0, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score
