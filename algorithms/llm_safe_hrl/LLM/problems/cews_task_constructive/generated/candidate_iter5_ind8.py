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
    Hybrid priority rule: combines Parent 2's robust relative urgency and risk-adaptive energy weighting
    with Parent 1's calibrated criticality-energy coupling, percentile-based waiting boost, and tight uncertainty gating.
    
    Key novelties:
    - Unified urgency: uses both relative slack ratio AND absolute exponential penalty for hard violations (slack < 0)
    - Criticality-energy score normalized via robust IQR *and* scaled by task importance tier (high/mid/low upward_rank)
    - Adaptive waiting boost bounded by work-deadline context (prevents starvation without over-dominance)
    - Uncertainty gating activated only under deadline pressure (slack < 30s), weighted by normalized uncertainty rank
    - All components eps-protected, finite-bounded, and shape-compliant; deterministic and side-effect-free.
    """
    eps = 1e-08
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

    # Robust normalization preserving sign and boundedness
    def robust_normalize(x):
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1 + eps
        med = np.median(x)
        z = (x - med) / iqr
        return np.clip(z, -10.0, 10.0)

    # --- Urgency: hybrid of relative slack & absolute violation penalty ---
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = slack / task_min_duration
    # Hard violation: exponential penalty for slack < 0
    abs_violation_penalty = np.where(slack < 0, np.clip(np.exp(-slack), 1.0, 25000.0), 1.0)
    # Soft urgency: inverse-linear for positive but tight slack (< 30s)
    soft_urgency = np.where((slack >= 0) & (slack < 30.0), 1.0 / (1.0 + 0.05 * slack + eps), 0.01)
    deadline_urgency = np.where(slack < 0, abs_violation_penalty, soft_urgency)

    # --- Risk-adaptive energy weighting (Parent 2 style) ---
    risk_exponent = np.clip(1.0 + 0.5 * np.maximum(0.0, -slack), 1.0, 3.0)
    energy_risk_weighted = min_incremental_energy * np.power(1.0 + uncertainty, risk_exponent)
    energy_safe = np.maximum(energy_risk_weighted, eps)

    # --- Criticality-energy efficiency: calibrated & tiered (Parent 1 + 2 fusion) ---
    crit_eff_ratio = upward_rank / energy_safe
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-6, 1e6)
    norm_crit_eff = robust_normalize(crit_eff_ratio)
    # Tiered scaling: amplify high-criticality tasks (top 33%), dampen low (bottom 33%)
    ur_quantiles = np.quantile(upward_rank, [0.33, 0.67])
    tier_factor = np.where(upward_rank > ur_quantiles[1], 1.3,
                          np.where(upward_rank < ur_quantiles[0], 0.7, 1.0))

    # --- Waiting boost: percentile-based & context-bounded (Parent 1 robustness + Parent 2 fairness) ---
    wait_rank = np.argsort(np.argsort(ready_wait_time))  # 0-indexed rank
    wait_percentile = wait_rank / (N - 1 + eps) if N > 1 else np.array([0.0])
    # Bounded boost: scaled by remaining_work (avoid boosting trivial long-wait tasks) and slack
    rw_med = np.median(remaining_work) + eps
    rw_norm = np.clip(remaining_work / rw_med, 0.1, 10.0)
    slack_sensitivity = np.clip(1.0 - np.maximum(0.0, slack) / (30.0 + eps), 0.0, 1.0)  # active when slack < 30s
    wait_boost = 0.25 * (1.0 / (1.0 + np.exp(-(wait_percentile * 10.0 - 5.0)))) * rw_norm * slack_sensitivity

    # --- Uncertainty gating: only under deadline pressure (Parent 1's tight mask + Parent 2's context) ---
    uncertainty_mask = (slack < 30.0).astype(float)
    norm_uncertainty = robust_normalize(uncertainty) * uncertainty_mask

    # --- Work & time cost normalization ---
    work_norm = robust_normalize(remaining_work)
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_normalize(time_cost)

    # --- Final score: prioritizes urgency first, then critical-efficiency, penalizes uncertainty and time ---
    # Coefficients tuned to reflect objective: minimize risk-adjusted energy subject to hard DDL
    score = (
        -4.0 * deadline_urgency          # highest weight: enforce deadline feasibility
        - 2.0 * tier_factor * norm_crit_eff  # reward high-criticality energy efficiency
        + 0.3 * time_norm                # mild penalty for long exec/comm time
        + 0.2 * work_norm                # slight penalty for large remaining work (indirectly favors parallelism)
        + 0.15 * norm_uncertainty        # penalty only when uncertainty matters (tight slack)
        + wait_boost                     # bounded starvation mitigation
    )

    # Ensure finite output and correct shape
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
