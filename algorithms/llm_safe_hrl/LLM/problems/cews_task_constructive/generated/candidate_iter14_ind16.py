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
    v2: Deadline-hardened, critical-path-aware, numerically robust priority scorer.
    Combines best practices from v0 and v1 with novel improvements:
    - Uses *slack-relative percentile gating* (v1) for energy efficiency, but refined to top-25% slack margin for tighter deadline focus.
    - Replaces sigmoid urgency with *arctan-based normalized urgency gradient* (v0-inspired, v1-enhanced): smooth, monotonic, scale-invariant penalty ramping for overdue tasks.
    - Introduces *critical-path pressure density*: (upward_rank * remaining_work) / (min_exec_time + min_comm_time + eps), capturing work-per-latency under DAG critical path.
    - Uncertainty penalty is *jointly gated* by both normalized urgency quantile (>70th) AND normalized uncertainty quantile (>70th), scaled linearly by absolute slack distance — penalizes high risk most when deadlines are tight or violated.
    - Fairness via *wait-time percentile* (v1), but multiplied by urgency-scaled weight to avoid starving late tasks.
    - Energy term includes *saturation guard* and *slack-aware sign reversal*: prioritizes low energy only when slack > 0; otherwise neutral or slightly penalizing for overdue tasks.
    - All normalizations use robust std fallback; final score strictly bounded and shape-enforced.
    """
    eps = 1e-08
    N = len(slack)
    
    # Input sanitization: ensure finite floats, no in-place mutation
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
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.array([0.0])
        std_x = np.std(x)
        if std_x < eps:
            return np.zeros_like(x)
        return (x - np.mean(x)) / (std_x + eps)
    
    # Urgency: smooth arctan-based gradient over normalized slack, monotonic & bounded
    norm_slack = robust_normalize(slack)
    urgency_raw = np.where(slack < 0, -slack, 0.0)  # absolute lateness risk
    urgency_arctan = np.arctan(-slack / (np.median(np.abs(slack)) + eps))  # scale-invariant, monotonic
    norm_urgency = robust_normalize(urgency_arctan)
    
    # Critical-path pressure density: work importance per unit latency
    base_latency = min_exec_time + min_comm_time + eps
    cp_pressure_density = (upward_rank * (remaining_work + eps)) / base_latency
    norm_cp_pressure = robust_normalize(cp_pressure_density)
    
    # Slack-relative percentile gating for energy efficiency (top-25% slack margin)
    slack_percentile = np.percentile(slack, 75)  # top 25% slack margin
    energy_gate = (slack >= slack_percentile).astype(float)
    # Energy efficiency: low incremental energy per work unit — but reverse sign for minimization
    energy_efficiency = -(min_incremental_energy + eps) / (remaining_work + eps)
    norm_energy_eff = robust_normalize(energy_efficiency)
    energy_term = norm_energy_eff * energy_gate
    energy_term = np.clip(energy_term, -4.0, 4.0)  # saturation guard
    
    # Uncertainty penalty: jointly gated by high urgency AND high uncertainty quantiles, scaled by |slack| proximity
    norm_uncert = robust_normalize(uncertainty)
    urg_quantile = np.quantile(urgency_raw, 0.7)
    uncert_quantile = np.quantile(norm_uncert, 0.7)
    # Penalize only when both urgency and uncertainty are high, and penalty grows as slack approaches zero or goes negative
    abs_slack = np.abs(slack) + eps
    slack_proximity = 1.0 / (abs_slack)  # higher penalty when slack near zero or negative
    uncertainty_penalty = np.where(
        (urgency_raw > urg_quantile) & (norm_uncert > uncert_quantile),
        uncertainty * slack_proximity,
        0.0
    )
    norm_uncert_penalty = robust_normalize(uncertainty_penalty)
    
    # Fairness: wait-time percentile weighted by urgency to prevent starvation of overdue tasks
    wait_rank = np.argsort(np.argsort(ready_wait_time)) / max(len(ready_wait_time) - 1, 1)
    fairness_term = -wait_rank * (1.0 + 0.5 * norm_urgency)  # urgency-boosted fairness
    
    # Latency risk term: imminent-window arctan volatility scaled by urgency
    mean_base_latency = np.mean(base_latency) + eps
    relative_volatility = base_latency * (uncertainty + eps) / mean_base_latency
    adaptive_window = np.maximum(0.3, 0.05 * mean_base_latency)
    imminent_window = (slack >= 0) & (slack < adaptive_window)
    risk_arctan = np.where(imminent_window, 0.5 + 1.0/np.pi * np.arctan(relative_volatility), 0.0)
    latency_risk_term = robust_normalize(risk_arctan)
    
    # Weighted combination — tuned for deadline hardness first, then energy
    w_urgency = 8.0
    w_cp_pressure = 3.5
    w_energy = 2.0
    w_uncert = 1.8
    w_fairness = 1.3
    w_risk = 2.2
    
    score = (
        w_urgency * norm_urgency +
        w_cp_pressure * norm_cp_pressure +
        w_energy * energy_term +
        w_uncert * norm_uncert_penalty +
        w_fairness * fairness_term +
        w_risk * latency_risk_term
    )
    
    # Final sanitization and shape enforcement
    score = np.nan_to_num(score, nan=1e6, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)
    
    # Ensure shape (N,)
    if score.ndim == 0:
        score = np.array([score])
    else:
        score = score.reshape(-1)
    
    # Pad or truncate to exactly N elements
    if score.shape[0] < N:
        score = np.pad(score, (0, N - score.shape[0]), constant_values=1e6)
    elif score.shape[0] > N:
        score = score[:N]
    
    return score
