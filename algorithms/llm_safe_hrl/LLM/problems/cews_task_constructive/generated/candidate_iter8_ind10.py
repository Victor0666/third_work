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
    Hybrid priority rule combining v0's urgency fidelity and v1's robust normalization & gating.
    Key innovations:
    - Deadline urgency: sigmoid with adaptive scale (median |slack|) + exponential penalty for negative slack
    - Energy term: slack-gated critical-energy density with MAD-normalized boost only when slack > 0
    - Critical-path awareness: upward_rank * (remaining_work / workflow_median_rw) normalized by MAD
    - Latency fairness: relative latency burden modulated by wait-time-sensitive tanh, bounded to [0, 0.15]
    - Uncertainty: active only in high-risk regime (0 < slack <= 2.5 AND uncertainty > 0.03), gated by slack-sigmoid
    - Aging: activated only under deadline pressure (slack < 1.5), using percentile rank among urgent tasks
    - All operations epsilon-protected; final score finite, deterministic, shape-(N,)
    """
    eps = 1e-08
    # Ensure float dtype and sanitize inputs
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=1e6, neginf=eps)

    def safe_mad_normalize(x):
        """MAD normalization with full robustness: handles N=1, constant arrays, outliers."""
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.zeros_like(x)
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - median_x) / mad
        return np.clip(normed, -3.0, 3.0)

    # --- Deadline Urgency: hybrid sigmoid + exponential penalty ---
    abs_slack = np.abs(slack)
    slack_scale = np.median(abs_slack) + eps
    # Sigmoid risk for positive/near-zero slack (steep near zero)
    sigmoid_risk = 1.0 / (1.0 + np.exp(-slack / slack_scale))
    # Exponential penalty for negative slack (lateness)
    exp_penalty = np.where(slack < 0, np.exp(-slack / (np.min(abs_slack) + eps)), 0.0)
    exp_penalty = np.clip(exp_penalty, 0.0, 1e6)
    deadline_score = safe_mad_normalize(sigmoid_risk + 0.3 * exp_penalty)

    # --- Energy Efficiency: slack-gated critical-energy density ---
    energy_denom = min_incremental_energy + eps
    base_density = (upward_rank + eps) * (remaining_work + eps) / energy_denom
    # Only activate energy optimization when slack permits (slack > 0)
    critical_energy_density = np.where(slack > 0.0, base_density, 0.0)
    critical_energy_norm = safe_mad_normalize(critical_energy_density)

    # --- Dynamic Critical-Path Boost ---
    median_rw = np.median(remaining_work) + eps
    cp_boost = upward_rank * (remaining_work / median_rw)
    cp_boost_norm = safe_mad_normalize(cp_boost)

    # --- Latency Fairness Boost ---
    total_latency = min_exec_time + min_comm_time + eps
    max_latency = np.max(total_latency) + eps
    latency_ratio = total_latency / max_latency
    # Modulate fairness by wait time only when slack is tight (< 2.0), preventing starvation of safe tasks
    wait_modulator = np.tanh(0.7 * ready_wait_time / (np.abs(slack) + 0.5))
    fairness_boost = np.clip(latency_ratio * (1.0 + 0.4 * wait_modulator), 0.0, 0.15)

    # --- Uncertainty Gating: focused on high-risk, low-margin regime ---
    uncertainty_gated = np.where(
        (slack > 0.0) & (slack <= 2.5) & (uncertainty > 0.03),
        uncertainty * (1.0 / (1.0 + np.exp(-(2.0 - slack)))),
        0.0
    )
    uncertainty_norm = safe_mad_normalize(uncertainty_gated)

    # --- Aging Term: percentile-based among urgent tasks only ---
    aging_term = np.zeros_like(slack)
    urgent_mask = slack < 1.5
    if np.any(urgent_mask):
        urgent_waits = ready_wait_time[urgent_mask]
        if urgent_waits.size > 0:
            if urgent_waits.size == 1:
                aging_term[urgent_mask] = 1.0
            else:
                sorted_urgent = np.sort(urgent_waits)
                for i in range(len(slack)):
                    if urgent_mask[i]:
                        # Percentile rank: fraction of urgent tasks with wait time <= current
                        pos = np.searchsorted(sorted_urgent, ready_wait_time[i], side='right')
                        aging_term[i] = pos / len(sorted_urgent)

    # --- Final weighted score: prioritizes deadline adherence first, then efficiency, then fairness ---
    score = (
        +5.0 * deadline_score           # Dominant deadline enforcement
        - 2.5 * critical_energy_norm   # Energy savings only when safe
        - 1.5 * cp_boost_norm          # Amplify critical path impact
        + 0.12 * fairness_boost        # Mild load balancing
        + 0.08 * aging_term            # Urgent task aging
        + 0.10 * uncertainty_norm      # Focused uncertainty mitigation
    )

    # Final sanitization: ensure finite, deterministic, shape-(N,)
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score.reshape(-1)
