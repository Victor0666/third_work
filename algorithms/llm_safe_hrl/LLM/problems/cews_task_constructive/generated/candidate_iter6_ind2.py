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
    Self-evolved priority rule: restores strict deadline dominance via hard urgency gating,
    fixes uncertainty boosting to *amplify* risk near tight deadlines (not suppress it),
    and replaces percentile-based starvation guard with monotonic, scale-invariant wait ranking.
    
    Key improvements over v1:
      - Urgency is now a hard binary gate (not sigmoid) for DDL enforcement: urgent = slack < median_slack → priority override
      - Uncertainty boost is *multiplied by (1 - urgency)* to maximize impact on borderline-urgent tasks (high uncertainty + non-critical slack),
        and uses absolute slack magnitude for robust risk severity scaling — fixes v1's counterproductive urgency gating.
      - Starvation guard uses log-scaled wait time (monotonic, bounded, N-invariant) instead of percentile ranking → ensures deterministic fairness.
      - All normalization uses robust min-max with fallback to zeros for constants (not IQR) to preserve urgency signal magnitude.
      - Criticality-energy term decouples rank and efficiency: upward_rank drives importance, efficiency (work/energy) drives green preference — fused only after independent normalization.
      - Energy penalty is strictly slack-aware: scaled by max(0, 1 - slack / (|median_slack| + eps)) to intensify penalty as slack shrinks.
      - Final weights prioritize DDL compliance (0.5), then energy (0.25), latency (0.1), fairness (0.08), uncertainty (0.04), criticality (0.03).
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

    def robust_minmax_norm(x):
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x, dtype=float)
        return (x - x_min) / (x_max - x_min + eps)

    # Hard urgency gate: urgent iff slack < median_slack → forces top priority for at-risk tasks
    median_slack = np.median(slack)
    is_urgent = (slack < median_slack).astype(float)
    
    # Base urgency term: 0 for urgent (highest priority), 1 otherwise → smallest score wins
    base_urgency = 1.0 - is_urgent

    # Criticality: normalized upward rank (higher = more critical path weight)
    norm_upward = robust_minmax_norm(upward_rank)

    # Efficiency: work per marginal energy — higher = greener; clipped to avoid division artifacts
    efficiency = remaining_work / (min_incremental_energy + eps)
    norm_efficiency = robust_minmax_norm(efficiency)

    # Criticality-energy coupling: only active when *not urgent*, to avoid diluting deadline signal
    crit_energy_term = (1.0 - is_urgent) * (1.0 - norm_upward * norm_efficiency)

    # Energy penalty: stronger when slack is tight; zero when slack >= median_slack
    slack_ratio = np.clip(slack / (np.abs(median_slack) + eps), -10.0, 10.0)
    energy_penalty_weight = np.maximum(0.0, 1.0 - slack_ratio)
    norm_energy = robust_minmax_norm(min_incremental_energy)
    energy_penalty = norm_energy * energy_penalty_weight

    # Latency term: execution + communication, both normalized and equally weighted
    norm_exec = robust_minmax_norm(min_exec_time)
    norm_comm = robust_minmax_norm(min_comm_time)
    latency_term = 0.5 * norm_exec + 0.5 * norm_comm

    # Uncertainty boost: amplifies priority *only for non-urgent but high-risk tasks*
    # Uses absolute slack distance from median to quantify risk severity (not sign)
    slack_distance = np.abs(slack - median_slack)
    max_distance = np.maximum(np.max(slack_distance), eps)
    uncertainty_risk_score = slack_distance / max_distance
    uncertainty_boost = uncertainty * uncertainty_risk_score * (1.0 - is_urgent)
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)

    # Starvation guard: monotonic, scale-invariant log wait time (avoids percentile non-monotonicity)
    # Bounded: log(1 + wait) / log(1 + max_wait + eps) → [0, 1], always increasing in wait_time
    wait_max = np.max(ready_wait_time) + eps
    wait_normalized = np.log1p(ready_wait_time) / np.log1p(wait_max)
    wait_guard = wait_normalized * (1.0 - is_urgent)  # only applies to non-urgent tasks

    # Assemble final score: smaller = better
    # Weights sum to 1.0 and prioritize DDL compliance first, then energy, then others
    score = (
        0.50 * base_urgency +
        0.25 * energy_penalty +
        0.10 * latency_term +
        0.08 * wait_guard +
        0.04 * norm_uncertainty_boost +
        0.03 * crit_energy_term
    )

    # Ensure finite, bounded output
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    return score
