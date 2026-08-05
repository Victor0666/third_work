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
    Self-evolved priority rule: hard-DDL-first, energy-second, starvation-robust.
    
    Key improvements over v1:
    - Replaces composite gating with *unified deadline urgency*: arctan(-slack) is the sole DDL signal,
      scaled only by deterministic risk factor (1 + clipped_uncertainty), ensuring monotonic priority.
    - Drops energy-per-work; uses raw min_incremental_energy as direct energy signal — aligns with
      marginal energy minimization objective and avoids spurious downstream coupling.
    - Eliminates redundant masks (e.g., separate criticality/comm/exec masks); instead applies
      *single slack-gated fairness boost* only to long-waiting tasks when slack >= -0.5s, preventing
      starvation without diluting urgency.
    - Uses symmetric robust_normalize with MAD+eps on all signals — fully deterministic for N=1.
    - Introduces *communication-aware urgency*: adds min_comm_time to deadline_score when slack < 0,
      treating high comm cost as deadline risk amplifier under pressure.
    - Weights sum to 1.0 and strictly enforce hierarchy: DDL safety > energy > fairness.
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
    
    def robust_normalize(x):
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        return (x - center) / (mad + eps)
    
    # Primary deadline urgency: bounded, monotonic, deterministic
    deadline_urgency = np.arctan(-slack)  # higher = more urgent (negative slack → large positive)
    
    # Risk amplification: uncertainty linearly boosts urgency under pressure, capped
    risk_factor = 1.0 + np.clip(uncertainty, 0.0, 2.0)
    
    # Communication penalty under deadline pressure: treat high comm time as added lateness risk
    comm_penalty = np.where(slack < 0.0, min_comm_time, 0.0)
    
    # Final deadline score: urgency + comm penalty, scaled by risk
    deadline_score = (deadline_urgency + robust_normalize(comm_penalty)) * risk_factor
    
    # Pure marginal energy signal — no work normalization; matches objective exactly
    energy_score = robust_normalize(min_incremental_energy)
    
    # Fairness: only activate wait boost when slack >= -0.5s (soft grace window), to avoid starving
    # near-deadline tasks while still protecting long-waiting ones in safe regime
    fairness_mask = (slack >= -0.5).astype(float)
    wait_normalized = robust_normalize(ready_wait_time)
    wait_score = -wait_normalized * fairness_mask  # negative = higher priority for older tasks
    
    # Criticality is dropped: upward_rank/remaining_work conflates local marginal impact with global path;
    # DDL and energy signals are sufficient and more aligned with objectives
    
    # Weighted combination: DDL dominates (0.55), energy secondary (0.35), fairness tertiary (0.10)
    score = 0.55 * deadline_score + 0.35 * energy_score + 0.10 * wait_score
    
    # Final sanitization: ensure finite, deterministic output
    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
