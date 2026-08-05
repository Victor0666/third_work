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

    '''
    v2 priority rule: Adaptive deadline-respectful urgency + criticality-aware risk-energy tradeoff +
                      uncertainty-modulated fairness + robust normalization.

    Key self-evolution improvements:
    - Replace hard violation penalty with *smooth urgency bias*: use clipped linear slack penalty
      (not binary) to preserve ranking continuity near deadline and avoid cascading starvation.
    - Lift criticality gating on energy optimization → apply *graded energy penalty* weighted by
      normalized upward_rank, enabling energy-aware scheduling even on medium-criticality paths.
    - Introduce *uncertainty-modulated fairness*: boost waiting tasks only when uncertainty is high
      AND wait_time exceeds duration-based threshold, avoiding spurious relief for trivial waits.
    - Add *work-intensity correction*: penalize low compute-density tasks (high comm/comp ratio)
      to reduce communication-induced energy waste and idle time.
    - Use *slack-aware energy density rescaling* with soft saturation: 1/(1 + |rel_slack|^0.5) 
      for stronger emphasis under tight slack without numerical explosion.
    - All norms use 5%-95% clipping; all divisions guarded; all NaN/inf replaced deterministically.
    '''
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

    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p05 = np.percentile(x, 5.0)
        p95 = np.percentile(x, 95.0)
        x_clipped = np.clip(x, p05, p95)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Duration and relative slack (guarded)
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)

    # Energy density with smooth slack rescaling: stronger penalty under tight slack
    energy_density = np.divide(min_incremental_energy, duration + eps, out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    slack_rescale = 1.0 + np.sqrt(np.abs(rel_slack) + eps)  # softer than linear, avoids blowup
    risk_adjusted_energy = energy_density / (slack_rescale + eps)

    # Criticality weight: graded, not gated — enables energy optimization across all ranks
    norm_upward_rank = robust_minmax_norm(upward_rank)
    # Work intensity: comm-heavy tasks hurt energy & latency → penalize low work_density
    work_density = np.divide(remaining_work, duration + eps, out=np.zeros_like(remaining_work), where=duration + eps != 0)
    norm_work_density = robust_minmax_norm(work_density)
    # Low work density → high comm/comp ratio → higher penalty
    work_intensity_penalty = 1.0 - norm_work_density

    # Fairness: uncertainty-modulated wait boost — only when wait > 1.5× duration AND uncertainty high
    wait_ratio = np.divide(ready_wait_time, duration + eps, out=np.zeros_like(ready_wait_time), where=duration + eps != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    norm_uncertainty = robust_minmax_norm(uncertainty)
    norm_ready_wait = robust_minmax_norm(ready_wait_time)
    fairness_boost_mask = (wait_ratio > 1.5).astype(float) * (norm_uncertainty > 0.7).astype(float)
    fairness_boost = norm_ready_wait * fairness_boost_mask

    # Urgency term: smooth linear penalty for negative slack, saturating at -0.5; zero for positive slack
    urgency_bias = np.clip(-rel_slack, 0.0, 0.5)  # maps [-∞,0] → [0,0.5], [0,+∞) → 0
    urgency_bias = robust_minmax_norm(urgency_bias)  # normalize for consistent scale

    # Base components (all normalized, finite, deterministic)
    norm_risk_energy = robust_minmax_norm(risk_adjusted_energy)
    norm_remaining_work = robust_minmax_norm(remaining_work)

    # Final score: lower = better
    # Prioritize urgency first, then energy-efficiency on critical paths, then fairness & work-intensity
    score = (
        0.38 * urgency_bias +
        0.24 * norm_risk_energy * (0.5 + 0.5 * norm_upward_rank) +  # graded energy penalty
        0.16 * (1.0 - fairness_boost) +  # boost → reduces score
        0.12 * work_intensity_penalty +
        0.10 * norm_remaining_work
    )

    # Apply strict numerical hygiene
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)

    # Ensure shape compliance
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
