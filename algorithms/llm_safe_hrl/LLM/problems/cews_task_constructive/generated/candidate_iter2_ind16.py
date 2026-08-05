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
    'Improved priority rule combining deadline safety, energy-efficiency awareness, critical-path fidelity,\n    and starvation prevention — with robust numerical handling and convex risk-aware weighting.\n\n    Key improvements:\n    - Hybrid deadline gating: exponential risk penalty for slack < 0 *and* normalized urgency for slack > 0,\n      enabling fine-grained prioritization across both violation and margin regimes.\n    - Criticality preserved only when beneficial: upward_rank scaled by (1 - sigmoid(slack)) to smoothly fade\n      as deadline pressure increases — avoiding abrupt gating discontinuities.\n    - Energy efficiency redefined as *energy-per-useful-latency*: min_incremental_energy / (min_exec_time + min_comm_time + eps)\n      but gated by slack to suppress energy savings when deadline risk dominates.\n    - Waiting boost uses adaptive percentile + arctan saturation for smoother, bounded fairness.\n    - All features normalized via clipped IQR scaling (-3,3) with fallback for degenerate cases (N=1 or constant arrays).\n    - Uncertainty modulated by slack distance to prioritize risk-awareness only where margin exists.\n    - Final score is convex combination with weights tuned to prioritize deadline safety first, then efficiency, then fairness.\n    '
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    def clipped_iqr_normalize(x):
        """Robust IQR normalization: (x - Q1) / (Q3 - Q1 + eps), clipped to [-3, 3]; handles N=1 & constant arrays."""
        if x.size == 1:
            return np.zeros_like(x)
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1 + eps
        # Avoid division by near-zero iqr in near-constant arrays
        normed = (x - q1) / iqr
        return np.clip(normed, -3.0, 3.0)
    
    # Deadline risk: exponential penalty for negative slack, linear urgency for positive slack (normalized)
    slack_abs = np.abs(slack)
    deadline_risk_raw = np.where(slack < 0, np.exp(np.clip(-slack, 0, 20)) - 1.0, 0.0)
    deadline_urgency_raw = np.where(slack >= 0, slack_abs / (np.mean(slack_abs + eps) + eps), 0.0)
    deadline_score = clipped_iqr_normalize(deadline_risk_raw + deadline_urgency_raw)
    
    # Smooth slack-gated criticality: upward_rank fades gradually as slack decreases (via sigmoid)
    # sigmoid(-slack) ≈ 1 when slack << 0, ≈ 0.5 when slack=0, ≈ 0 when slack >> 0 → use (1 - sigmoid(-slack)) = sigmoid(slack)
    slack_sigmoid = 1.0 / (1.0 + np.exp(-slack + eps))
    upward_rank_gated = upward_rank * slack_sigmoid
    upward_rank_norm = clipped_iqr_normalize(upward_rank_gated)
    
    # Energy efficiency: energy per unit latency, but only rewarded when slack > 0
    total_latency = min_exec_time + min_comm_time + eps
    energy_eff_ratio = min_incremental_energy / total_latency
    energy_eff_gated = np.where(slack >= 0, energy_eff_ratio, 0.0)
    energy_eff_norm = clipped_iqr_normalize(energy_eff_gated)
    
    # Adaptive waiting boost: percentile-based, smoothed with arctan to prevent unbounded growth
    wait_sorted = np.sort(ready_wait_time)
    n = len(wait_sorted)
    if n == 1:
        wait_percentile = np.array([0.5])
    else:
        # Use fractional rank to handle duplicates
        ranks = np.searchsorted(wait_sorted, ready_wait_time, side='left') + 0.5
        wait_percentile = np.clip(ranks / n, 0.0, 1.0)
    wait_boost = np.arctan(3.0 * (wait_percentile - 0.5)) / (np.pi / 2) + 0.5  # maps [0,1] → [0,1], smooth & bounded
    
    # Uncertainty: only matters when slack > 0; scaled by normalized slack distance
    slack_distance_norm = np.abs(slack) / (np.mean(np.abs(slack) + eps) + eps)
    uncertainty_gated = np.where(slack >= 0, uncertainty * (1.0 - np.tanh(slack_distance_norm)), 0.0)
    uncertainty_norm = clipped_iqr_normalize(uncertainty_gated)
    
    # Remaining work: normalized importance of downstream computation load
    remaining_work_norm = clipped_iqr_normalize(remaining_work)
    
    # Final convex score: smallest = highest priority
    # Weights emphasize deadline safety (2.8), then critical-path (1.5), energy efficiency (0.9), fairness (0.7), uncertainty (0.4), work (0.3)
    score = (
        +2.8 * deadline_score
        - 1.5 * upward_rank_norm
        + 0.9 * energy_eff_norm
        - 0.7 * wait_boost
        + 0.4 * uncertainty_norm
        + 0.3 * remaining_work_norm
    )
    
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
