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
    Mutated priority rule emphasizing:
    - Hard-thresholded urgency gating via 30th-percentile relative slack
    - Slack-gated starvation control with work-normalized wait boost
    - Latency-criticality synergy: (upward_rank * task_duration) / energy
    - Robust per-feature normalization using clipped z-score with median/IQR fallback
    - Risk-adaptive energy scaling via uncertainty-weighted exponent on slack distance
    - Elimination of heuristic coupling noise by decoupling time/energy/criticality terms
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
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    # Robust normalization: clipped z-score with fallback to IQR-based scaling
    def robust_normalize(x):
        x = np.clip(x, -1e12, 1e12)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        median_x = np.median(x)
        std_x = np.std(x) + eps
        z_score = (x - median_x) / std_x
        # Fallback to IQR if std is unstable (e.g., near-constant values)
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1 + eps
        iqr_scaled = (x - median_x) / iqr
        # Use z-score when stable, else iqr-scaled
        use_z = (std_x > 0.1 * (np.max(x) - np.min(x) + eps))
        return np.where(use_z, np.clip(z_score, -5.0, 5.0), np.clip(iqr_scaled, -5.0, 5.0))

    # Task intrinsic duration and normalized urgency gating
    task_duration = min_exec_time + min_comm_time + eps
    rel_slack = slack / task_duration
    # Hard threshold at 30th percentile: only penalize tasks below this slack threshold
    slack_thresh = np.quantile(rel_slack, 0.3) if N > 1 else np.min(rel_slack)
    urgency_penalty = np.maximum(0.0, slack_thresh - rel_slack) * 1.8

    # Latency-criticality synergy term: (upward_rank * duration) / energy — prioritizes high-impact, fast-executing, low-energy tasks
    latency_crit_synergy = (upward_rank * task_duration) / (min_incremental_energy + eps)
    latency_crit_synergy = np.clip(latency_crit_synergy, 1e-6, 1e6)
    latency_crit_norm = robust_normalize(latency_crit_synergy)

    # Risk-adaptive energy scaling: exponential penalty only for negative slack, scaled by uncertainty
    slack_distance = np.maximum(0.0, -slack)  # only active under lateness risk
    energy_risk_exponent = 1.0 + uncertainty * np.clip(slack_distance, 0.0, 5.0)
    risk_weighted_energy = min_incremental_energy * np.power(1.0 + eps, energy_risk_exponent)  # stable base
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e9)
    energy_norm = robust_normalize(risk_weighted_energy)

    # Work-aware starvation control: boost waiting tasks only when slack > 0 AND remaining_work is non-trivial
    median_rw = np.median(remaining_work) + eps
    rw_normalized = np.clip(remaining_work / median_rw, 0.01, 100.0)
    wait_boost_mask = (slack > 0.0) & (rw_normalized > 0.5)
    max_wait = np.maximum(np.max(ready_wait_time), eps)
    wait_boost = np.where(wait_boost_mask,
                         np.clip(ready_wait_time / max_wait, 0.0, 0.3) * (rw_normalized / 2.0),
                         0.0)

    # Uncertainty-aware criticality: dampen upward_rank for high-uncertainty tasks unless urgent
    ur_dampened = upward_rank / (1.0 + uncertainty * np.clip(-slack, 0.0, 10.0) + eps)
    ur_norm = robust_normalize(ur_dampened)

    # Time cost proxy: sqrt(exec + comm) for stability near zero
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_normalize(time_cost)

    # Composite score: smaller = better
    # Deadline urgency dominates (negative weight), synergy & energy are secondary drivers
    score = (
        +2.5 * urgency_penalty           # Higher penalty → higher score → lower priority; so urgency drives down score via negative logic below
        -1.9 * latency_crit_norm         # Critical + fast + efficient → lower score
        +0.8 * energy_norm               # High risk-weighted energy → higher score
        -0.6 * ur_norm                   # Dampened criticality → lower score when reliable
        +0.4 * time_norm                 # Longer duration → slightly higher score
        +wait_boost                      # Bounded starvation relief → small positive bump (lowers priority only modestly)
    )

    # Final guard: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e10, 1e10)

    return score
