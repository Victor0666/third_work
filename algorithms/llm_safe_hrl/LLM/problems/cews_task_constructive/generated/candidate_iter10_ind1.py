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
    - Hard-thresholded urgency gating via 30th-percentile relative slack cutoff
    - Robust per-feature normalization using clipped z-score with median/IQR fallback
    - Risk-adaptive energy scaling where penalty exponent depends on slack quantile distance
    - Latency-criticality synergy term: (upward_rank * (min_exec_time + min_comm_time)) / (min_incremental_energy + eps)
    - Slack-gated starvation control: wait boost only for tasks above 30th-percentile slack AND below median work
    - Explicit dominance prevention: all components bounded and sign-aligned to ensure smaller = better
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

    # Robust normalization: clipped z-score with fallback to IQR-based scaling for skewed distributions
    def robust_normalize(x):
        median_x = np.median(x)
        std_x = np.std(x) + eps
        z_score = (x - median_x) / std_x
        # Fall back to IQR normalization if std is unstable (e.g., near-constant x)
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1 + eps
        iqr_norm = (x - median_x) / iqr
        # Use z-score unless it's extreme; cap both
        norm = np.where(np.abs(z_score) > 10.0, iqr_norm, z_score)
        return np.clip(norm, -8.0, 8.0)

    # --- Urgency gating: hard threshold at 30th-percentile relative slack ---
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = slack / task_min_duration
    slack_30p = np.quantile(rel_slack, 0.30)
    # Tasks with rel_slack <= 30th percentile get strong urgency boost (smaller score)
    urgency_gate = (rel_slack <= slack_30p).astype(float)
    # Urgency score: linear penalty for negative rel_slack, capped; zero otherwise
    urgency_linear = np.clip(-rel_slack, 0.0, 5.0) * urgency_gate
    # Normalize urgency component to [-1, 0] range so smaller = more urgent
    urgency_score = -np.clip(urgency_linear, 0.0, 1.0)

    # --- Risk-adaptive energy scaling ---
    # Exponent increases as slack falls below median slack (i.e., higher lateness risk)
    slack_median = np.median(slack) + eps
    slack_ratio = np.clip((slack_median - slack) / (np.abs(slack_median) + eps), 0.0, 4.0)
    risk_exponent = np.clip(1.0 + 0.7 * slack_ratio, 1.0, 3.5)
    energy_risk_weighted = min_incremental_energy * np.power(1.0 + uncertainty, risk_exponent)
    energy_safe = np.maximum(energy_risk_weighted, eps)
    # Energy efficiency: higher upward_rank / energy => better => lower score
    energy_efficiency = upward_rank / energy_safe
    energy_efficiency = np.clip(energy_efficiency, 1e-6, 1e6)
    energy_norm = robust_normalize(energy_efficiency)

    # --- Latency-criticality synergy: importance × duration per unit energy ---
    # Captures "bang-for-buck" in terms of critical path delay vs energy cost
    latency_crit = (upward_rank * (min_exec_time + min_comm_time)) / (energy_safe + eps)
    latency_crit = np.clip(latency_crit, 1e-6, 1e6)
    latency_norm = robust_normalize(latency_crit)

    # --- Slack-gated starvation control ---
    # Only activate wait boost for non-urgent tasks (rel_slack > 30th percentile) AND low-work tasks
    work_median = np.median(remaining_work) + eps
    low_work_mask = (remaining_work < work_median).astype(float)
    non_urgent_mask = (rel_slack > slack_30p).astype(float)
    wait_activation = non_urgent_mask * low_work_mask
    max_wait = np.maximum(np.max(ready_wait_time), eps)
    wait_normalized = np.clip(ready_wait_time / max_wait, 0.0, 1.0)
    # Bounded wait penalty: [0, 0.15], scaled by activation mask
    wait_penalty = wait_normalized * wait_activation * 0.15

    # --- Additional normalized features ---
    # Communication-to-compute ratio (indicates data-intensive bottlenecks)
    comm_compute_ratio = np.clip(min_comm_time / (min_exec_time + eps), 0.01, 100.0)
    comm_norm = robust_normalize(comm_compute_ratio)

    # Uncertainty-aware duration: penalize high-uncertainty long-duration tasks
    dur_uncertain = (min_exec_time + min_comm_time) * (1.0 + uncertainty)
    dur_uncertain_norm = robust_normalize(dur_uncertain)

    # Work-normalized energy: energy per MI — lower is better
    work_energy_ratio = energy_safe / (remaining_work + eps)
    work_energy_norm = robust_normalize(work_energy_ratio)

    # --- Final score assembly: all terms aligned so smaller = better ---
    # Weighting tuned to prioritize urgency > criticality-efficiency > fairness > robustness
    score = (
        +1.8 * urgency_score          # Strongest weight: hard deadline gate
        -1.4 * energy_norm           # Higher efficiency → lower score
        -1.1 * latency_norm          # Higher latency-criticality synergy → lower score
        +0.3 * comm_norm             # Higher comm/compute → slightly less preferred (edge-aware)
        +0.25 * dur_uncertain_norm   # Higher uncertain duration → slight penalty
        +0.2 * work_energy_norm      # Higher energy per MI → penalty
        +wait_penalty                # Small fairness boost when safe to apply
    )

    # Ensure finite output and deterministic shape
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
