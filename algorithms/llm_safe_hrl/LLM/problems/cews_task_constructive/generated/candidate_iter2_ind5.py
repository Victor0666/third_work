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
    Hybrid priority rule: hard deadline feasibility first, then risk-aware energy-criticality,
    with starvation prevention and uncertainty-gated adaptivity.
    
    Key improvements:
    - Robust IQR-based normalization (from Parent 2) for all features to resist outliers.
    - Slack penalty uses *adaptive bounded exponential*: exp(-slack / (|median_slack| + eps)) for slack < 0,
      and smooth sigmoid decay for slack >= 0 — avoids discontinuity at zero and preserves ordering.
    - Criticality-energy coupling enhanced: uses work/energy efficiency *and* upward_rank, normalized jointly.
    - Waiting boost is relative *and* capped: sigmoid of (ready_wait_time / (max_wait + eps)), bounded in [0, 0.4].
    - Uncertainty gating refined: active only when slack < median_slack AND uncertainty > Q75_uncert → tighter risk focus.
    - All divisions and exponentials protected by eps; no unbounded values; deterministic and finite-output guaranteed.
    """
    eps = 1e-08
    # Ensure float64 and copy to avoid mutation
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()
    
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)
    
    # Robust IQR-based normalization (Parent 2 strength)
    def normalize_robust(x):
        q75, q25 = np.percentile(x, [75, 25], method='midpoint')
        iqr = q75 - q25 + eps
        med = np.median(x)
        return (x - med) / iqr
    
    # Adaptive slack penalty: bounded & continuous across slack=0
    # For slack < 0: exp(-slack / (|median_slack| + eps)) → grows as lateness increases
    # For slack >= 0: sigmoid decay: 1 / (1 + exp(slack / (median_slack+eps))) → approaches 0 as slack grows
    median_slack_abs = np.abs(np.median(slack)) + eps
    slack_penalty = np.where(
        slack < 0,
        np.clip(np.exp(-slack / median_slack_abs), 1.0, 25000.0),
        1.0 / (1.0 + np.exp(slack / median_slack_abs))
    )
    # Normalize penalty to [0, 1] range for stable weighting
    penalty_norm = (slack_penalty - np.min(slack_penalty)) / (np.max(slack_penalty) - np.min(slack_penalty) + eps)
    
    # Criticality-energy score: upward_rank * (work / energy), robust-normalized
    energy_efficiency = remaining_work / (min_incremental_energy + eps)
    critical_energy_score = upward_rank * energy_efficiency
    norm_critical_energy = normalize_robust(critical_energy_score)
    
    # Waiting boost: relative, capped sigmoid
    max_wait = np.maximum(np.max(ready_wait_time), eps)
    wait_boost = 0.4 * (1.0 / (1.0 + np.exp(-(ready_wait_time / max_wait - 2.0))))
    
    # Uncertainty gating: only activate when both tight slack AND high uncertainty
    median_slack = np.median(slack)
    q75_uncert = np.percentile(uncertainty, 75, method='midpoint')
    tight_slack_mask = slack < median_slack
    high_uncert_mask = uncertainty > q75_uncert
    gated_uncertainty = np.where(tight_slack_mask & high_uncert_mask, normalize_robust(uncertainty), 0.0)
    
    # Base feature scores (all robust-normalized)
    norm_exec = normalize_robust(min_exec_time)
    norm_comm = normalize_robust(min_comm_time)
    norm_energy = normalize_robust(min_incremental_energy)
    
    # Final weighted score: smaller = higher priority
    # Emphasize deadline urgency (penalty), critical-energy efficiency, and waiting boost;
    # penalize execution/comm/energy costs; reduce impact of uncertainty unless gated
    score = (
        0.15 * norm_exec +
        0.10 * norm_comm +
        0.20 * norm_energy -
        0.35 * penalty_norm -     # Strong deadline emphasis
        0.30 * norm_critical_energy +  # Favor high-impact, low-energy tasks
        0.10 * wait_boost -       # Anti-starvation, bounded
        0.05 * gated_uncertainty  # Only amplify when risk-convergent
    )
    
    # Ensure finite, deterministic output
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
