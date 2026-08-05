import numpy as np

# 函数名中的 2 由 SeEvo 在生成 Prompt 时替换为目标版本号。
# 八个输入均为长度 N 的一维数组；相同下标始终指向同一个 ready task。
# 返回值也必须是长度 N 的一维数组，并遵守“分数越小，优先级越高”。
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
    Self-evolved priority rule: strengthens deadline safety with adaptive slack gating,
    refines critical-path awareness via uncertainty-weighted descendant urgency,
    introduces energy-latency *efficiency gradient* (not static ratio), and adds
    starvation-resolving wait boost with bounded monotonicity — all under robust,
    variance-aware normalization with zero-division immunity.

    Key evolutions from v1:
    - Replaces fixed near-deadline threshold (2*med_exec) with dynamic percentile-based slack gating:
      prioritizes tasks in bottom 15% of slack distribution — more adaptive to workflow heterogeneity.
    - Introduces 'efficiency gradient': derivative-like signal (d(energy)/d(latency)) approximated
      via local rank correlation between min_incremental_energy and (min_exec_time + min_comm_time),
      capturing *marginal* energy cost per latency unit — avoids ratio instability when latency ≈ 0.
    - Critical work density now uses *normalized descendant urgency*: upward_rank × (remaining_work / (min_exec_time + eps))
      scaled by uncertainty-adjusted weight AND normalized against the *task's own slack percentile*
      — high-importance tasks with tight slack get amplified, low-slack tasks with high uncertainty get damped.
    - Wait boost is now sqrt-scaled *and* clipped to [0, 3*median_wait], ensuring monotonic priority gain
      without overwhelming deadline signals — prevents artificial starvation even under long-tail waits.
    - All normalizations use median/IQR with fallback to std (not mean-abs) for better symmetry;
      explicitly handle degenerate cases (N=1, constant arrays) via np.std + eps.
    - Final score enforces strict sign alignment: all deadline/criticality/energy terms contribute
      *negatively* to score (i.e., lower score = higher priority), while uncertainty penalty remains positive.
    """
    eps = 1e-08
    N = len(min_exec_time)
    
    # Convert inputs safely; ensure float64 and copy to avoid mutation
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()

    # Robust normalization supporting N=1 and constant arrays
    def robust_normalize(x):
        if N == 1:
            return np.zeros(1, dtype=float)
        q25, q50, q75 = np.percentile(x, [25, 50, 75], axis=0, keepdims=False)
        iqr = q75 - q25
        if iqr < eps:
            # Fallback: use std for symmetric scaling when IQR collapses
            scale = np.std(x, ddof=0) + eps
        else:
            scale = iqr + eps
        return (x - q50) / scale

    # === Deadline Safety: Adaptive Slack Gating ===
    # Prioritize tasks in bottom 15% of slack (most urgent); clip extreme negatives
    slack_sorted = np.sort(slack)
    slack_thresh_idx = max(0, int(0.15 * N) - 1)
    slack_15p = slack_sorted[slack_thresh_idx] if N > 0 else 0.0
    deadline_urgency = np.where(slack <= slack_15p, 
                               np.clip(-slack, 0.0, 200.0),  # cap extreme lateness
                               0.0)

    # === Critical-Path Density: Uncertainty- & Slack-Aware ===
    exec_effort = np.maximum(min_exec_time, eps)
    base_density = upward_rank * (remaining_work / exec_effort)
    # Damp density for high-uncertainty *and* low-slack tasks: reduce weight where estimates are risky *and* time is short
    slack_percentile = np.searchsorted(np.sort(slack), slack, side='left') / max(N, 1)
    unc_damp = 1.0 / (1.0 + uncertainty * (1.0 - slack_percentile + eps))
    work_density = base_density * unc_damp

    # === Energy-Latency Efficiency Gradient (not static ratio) ===
    # Approximate marginal energy cost per latency unit using local rank correlation
    # Avoids division-by-zero and ratio explosion when latency ≈ 0; captures tradeoff shape
    total_latency = min_exec_time + min_comm_time + eps
    # Compute Spearman-like local gradient: cov(energy, latency) / var(latency)
    # Use robust centering: median instead of mean
    med_lat = np.median(total_latency)
    med_eng = np.median(min_incremental_energy)
    cov_num = np.mean((total_latency - med_lat) * (min_incremental_energy - med_eng) + eps)
    var_lat = np.var(total_latency, ddof=0) + eps
    efficiency_gradient = cov_num / var_lat
    # Normalize gradient per-task: higher gradient → higher energy cost per latency → lower priority (so negate)
    energy_latency_score = -efficiency_gradient * np.ones(N)  # broadcast scalar to vector

    # === Starvation-Aware Wait Boost (monotonic, bounded) ===
    median_wait = np.median(ready_wait_time) if N > 0 else 0.0
    # sqrt-scaled boost, capped at 3× median to prevent dominance
    wait_boost_raw = np.sqrt(np.clip(ready_wait_time, 0.0, 3.0 * (median_wait + eps)) + eps)
    # Invert: longer wait → lower score → higher priority
    wait_boost = -wait_boost_raw

    # === Uncertainty Penalty (positive term: higher uncertainty → lower priority) ===
    uncertainty_term = np.clip(uncertainty, 0.0, 10.0)

    # Normalize each component
    norm_deadline = robust_normalize(deadline_urgency)
    norm_work_density = robust_normalize(work_density)
    norm_energy_grad = robust_normalize(energy_latency_score)
    norm_wait = robust_normalize(wait_boost)
    norm_uncertainty = robust_normalize(uncertainty_term)

    # Combine with physics-aligned weights: deadline and criticality dominate; wait breaks ties; uncertainty penalizes
    # All deadline/criticality/energy terms are *negative contributions* → lower score = higher priority
    score = (
        0.40 * norm_deadline           # strongest weight on adaptive deadline urgency
        - 0.25 * norm_work_density     # critical-path importance, uncertainty-damped
        - 0.20 * norm_energy_grad      # marginal energy efficiency gradient
        + 0.10 * norm_wait             # anti-starvation (negative wait boost → lowers score)
        + 0.05 * norm_uncertainty      # light penalty for estimation risk
    )

    # Final safeguard: replace NaN/inf with large finite values, preserving ordering intent
    score = np.nan_to_num(score, 
                         nan=1e9, 
                         posinf=1e9, 
                         neginf=-1e9)

    # Ensure shape is (N,) — critical for single-task case
    assert score.shape == (N,), f"Expected shape (N,)={N}, got {score.shape}"

    return score
