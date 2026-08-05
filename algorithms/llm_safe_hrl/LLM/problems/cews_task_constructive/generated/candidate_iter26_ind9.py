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
    v2 priority rule: Hard-deadline dominance + slack-proximity energy gating + 
                      critical-path fidelity + fairness-aware starvation relief +
                      uncertainty-gated risk coupling + robust minmax normalization.
    
    Key synthesis:
    - Retains Parent 2's stable min-max normalization (handles N=1, skewed dists).
    - Integrates Parent 1's strict lateness penalty (unclipped, high-magnitude) for hard DDL enforcement.
    - Combines Parent 2's slack-proximity penalty (smooth, bounded) with Parent 1's urgency-aware wait relief.
    - Uses Parent 2's dual-gated uncertainty coupling (slack > 0 AND upward_rank > 75th percentile).
    - Replaces brittle median-based gates with percentile-based ones for better distribution robustness.
    - Adds normalized critical-path term (duration * upward_rank) weighted higher than energy to prioritize deadline feasibility.
    - Introduces *remaining-work fairness* term: penalizes long-wait tasks only when they carry significant remaining work.
    - All operations guarded; no NaN/inf/infinite loops; deterministic and side-effect-free.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)

    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0, method='lower') if N > 1 else np.min(x)
        p99 = np.percentile(x, 99.0, method='higher') if N > 1 else np.max(x)
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # 1. HARD DEADLINE ENFORCEMENT: unclipped massive penalty for slack <= 0
    lateness_penalty = np.where(slack <= 0.0, -1e12 + 1000.0 * np.abs(slack), 0.0)

    # 2. URGENCY & TIMING: duration-weighted critical path importance
    duration = min_exec_time + min_comm_time + eps
    critical_timing = duration * upward_rank
    norm_critical_timing = robust_minmax_norm(critical_timing)

    # 3. ENERGY-RISK: slack-proximity gated energy density penalty
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_minmax_norm(energy_density)
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    slack_proximity = np.clip(0.3 - rel_slack, 0.0, 0.3)  # active only for tight-but-feasible tasks
    energy_penalty = norm_energy_density * slack_proximity

    # 4. FAIRNESS: wait-time relief gated by both slack > 0 AND significant remaining work
    # Avoids biasing trivial tasks — only helps starving tasks that contribute meaningfully to workflow progress
    work_threshold = np.percentile(remaining_work, 75.0) + eps if N > 1 else np.max(remaining_work)
    fairness_mask = (slack > 0.0) & (remaining_work >= work_threshold)
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    fairness_boost = norm_wait_time * fairness_mask

    # 5. UNCERTAINTY COUPLING: gated by slack > 0 AND high criticality (top 25% upward_rank)
    rank_75 = np.percentile(upward_rank, 75.0) + eps if N > 1 else np.max(upward_rank)
    unc_mask = (slack > 0.0) & (upward_rank >= rank_75)
    norm_uncertainty = robust_minmax_norm(uncertainty)
    unc_coupling = norm_uncertainty * unc_mask

    # 6. FINAL SCORE: weighted sum with DDL feasibility dominant
    # Weights sum to 1.0 and reflect optimization hierarchy: DDL first, then timing, energy, fairness, risk
    score = (
        0.45 * norm_critical_timing +      # prioritizes critical path completion under deadline
        0.25 * robust_minmax_norm(remaining_work) +  # favors high-work tasks early (load balancing)
        0.18 * energy_penalty +           # applies energy cost only where it matters (tight slack)
        0.08 * fairness_boost +           # relieves starvation only for meaningful work
        0.04 * unc_coupling               # adds risk awareness only when safe and critical
    )

    # Apply hard deadline penalty last — dominates all other terms
    score = lateness_penalty + score

    # Final sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
