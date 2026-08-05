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
    Hybrid priority rule: hard-DDL safety first (additive offset + monotonic urgency),
    energy efficiency second (context-gated & critical-path normalized), fairness third.
    Combines v1's robust normalization and v2's strict DDL enforcement, adds:
      - Unified uncertainty coupling: amplifies urgency *only* when both slack<0 AND uncertainty>median,
        scaled by sigmoid of normalized urgency to avoid binary gating;
      - Critical-energy synergy: replaces energy_per_work with (upward_rank * remaining_work) / (energy+eps),
        weighted by slack sign and normalized separately for discriminative efficiency signal;
      - Bounded aging boost: tanh-based wait boost gated by slack margin *and* normalized wait percentile,
        ensuring fairness without compromising deadlines;
      - All components clipped and normalized per-array with explicit N=1 handling;
      - Final score strictly finite, deterministic, and prioritizes smaller values.
    """
    eps = 1e-8
    # Ensure float arrays and sanitize inf/nan
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e6, neginf=1e6)

    def robust_normalize(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        q25, q75 = np.percentile(x, [25, 75], overwrite_input=False)
        iqr = q75 - q25
        center = np.median(x, overwrite_input=False)
        if iqr > eps:
            scale = iqr
        else:
            std_val = np.std(x, ddof=0)
            scale = std_val if std_val > eps else np.mean(np.abs(x - center)) + eps
        norm = (x - center) / (scale + eps)
        return np.clip(norm, -3.0, 3.0)

    # Hard-DDL urgency: monotonic, bounded, additive offset for negative slack
    abs_slack = np.abs(slack)
    inv_slack = 1.0 / (abs_slack + eps)
    urgency_raw = np.where(slack < 0, inv_slack, 0.0)
    urgency_raw = np.clip(urgency_raw, 0.0, 1e3)
    norm_urgency = robust_normalize(urgency_raw)
    hard_ddl_offset = np.where(slack < 0, 200.0, 0.0)  # Strict dominance for missed deadlines

    # Critical-path-aware energy efficiency: higher rank & work → prioritize energy saving more
    crit_energy_score = (upward_rank * remaining_work) / (min_incremental_energy + eps)
    crit_energy_score = np.clip(crit_energy_score, 0.0, 1e9)
    # Gate energy signal only when slack >= 0; otherwise suppress to avoid distraction
    energy_weight_mask = (slack >= 0).astype(float)
    norm_crit_energy = robust_normalize(crit_energy_score) * energy_weight_mask

    # Uncertainty coupling: amplifies urgency only under stress (slack<0 AND high uncertainty)
    median_uncert = np.median(uncertainty) + eps
    risk_active = (slack < 0) & (uncertainty > median_uncert)
    # Sigmoid scaling ensures smooth, bounded amplification proportional to urgency level
    urgency_sigmoid = 1.0 / (1.0 + np.exp(-(norm_urgency + 1.0)))  # shifted to emphasize urgency
    uncertainty_penalty = np.where(risk_active, uncertainty * urgency_sigmoid, 0.0)
    norm_uncert_penalty = robust_normalize(uncertainty_penalty)

    # Aging boost: tanh-based wait boost, gated by both slack margin and percentile rank
    median_wait = np.median(ready_wait_time) + eps
    wait_tanh = np.tanh(ready_wait_time / (median_wait + eps))
    # Normalize wait rank to prevent dominance when N=1 or uniform waits
    if len(ready_wait_time) > 1:
        wait_rank = np.argsort(np.argsort(ready_wait_time)) / (len(ready_wait_time) - 1 + eps)
        wait_boost = wait_tanh * wait_rank
    else:
        wait_boost = np.array([0.0])
    # Gate by slack margin: only activate fairness when safe
    slack_margin = np.clip(slack, 0.0, 1e6)
    tau_safe = np.clip(np.mean(slack_margin) + eps, eps, 1e3)
    slack_safety_ratio = np.clip(slack_margin / (tau_safe + eps), 0.0, 1.0)
    wait_boost = wait_boost * slack_safety_ratio
    norm_wait_boost = robust_normalize(wait_boost)

    # Execution & communication cost signals (lower is better, so invert after normalization)
    norm_exec = robust_normalize(min_exec_time)
    norm_comm = robust_normalize(min_comm_time)

    # Weighted combination: urgency dominates, then critical path, energy, uncertainty, fairness
    w_urgency = 6.0
    w_critical = 1.8  # upward_rank * (min_exec_time + min_comm_time) captures critical path footprint
    w_energy = 2.2
    w_uncert = 1.1
    w_wait = 0.6
    w_exec = 0.15
    w_comm = 0.08

    # Critical density for scheduling pressure: higher means more time-sensitive
    latency_footprint = np.maximum(min_exec_time + min_comm_time, eps)
    critical_density = upward_rank * latency_footprint
    norm_critical_density = robust_normalize(critical_density)

    score = (
        w_urgency * norm_urgency +
        w_critical * norm_critical_density +
        w_energy * norm_crit_energy +
        w_uncert * norm_uncert_penalty +
        w_wait * norm_wait_boost +
        w_exec * norm_exec +
        w_comm * norm_comm +
        hard_ddl_offset
    )

    # Final sanitization: ensure finite, deterministic, shape-(N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score
