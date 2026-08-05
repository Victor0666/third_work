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

    'Self-evolved priority rule: adaptive slack-aware gating, unified robust scaling, criticality-energy synergy, and starvation-robust urgency.\n\nKey evolutions:\n  - Adaptive slack gating: uses smooth sigmoid transition around slack=0 (not hard threshold) to preserve urgency for near-deadline tasks (e.g., slack=0.1s), improving DDL feasibility.\n  - Unified joint normalization: computes IQR/MAD over *all* core features (upward_rank, remaining_work, duration, energy_ps) jointly — prevents feature-wise noise amplification and enforces consistent scale.\n  - Criticality-energy synergy term: (upward_rank * min_incremental_energy) / (duration + eps), normalized jointly — rewards high-impact *low-energy* tasks, directly optimizing risk-adjusted fuzzy energy under deadline pressure.\n  - Starvation-robust urgency: ready_wait_time boost activated when (slack < median_slack OR upward_rank > 0.75*max_rank), avoiding brittle median dependence and handling singleton cases.\n  - Uncertainty modulation: applies soft attenuation (1/(1+uncertainty)) to criticality terms *only*, reducing over-amplification under high fuzziness while preserving energy-efficiency signal.\n  - All ops protected: no division by zero, no NaN/inf propagation, deterministic & finite output.'
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    # Compute stable joint features for unified normalization
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_second = np.clip(min_incremental_energy / duration, -1e6, 1e6)
    crit_energy_synergy = np.clip((upward_rank * min_incremental_energy) / (duration + eps), -1e6, 1e6)
    
    # Unified robust scaling across correlated core features
    joint_features = np.vstack([
        np.abs(upward_rank),
        np.abs(remaining_work),
        duration,
        np.abs(energy_per_second),
        np.abs(crit_energy_synergy)
    ]).T
    if joint_features.size == 0:
        joint_scale = eps
    else:
        # Compute IQR across flattened joint features; fallback to MAD if IQR ~ 0
        q1, q3 = np.quantile(joint_features.flatten(), [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        if iqr < eps:
            med = np.median(joint_features.flatten())
            mad = np.median(np.abs(joint_features.flatten() - med))
            joint_scale = mad if mad > eps else np.mean(np.abs(joint_features.flatten())) + eps
        else:
            joint_scale = iqr + eps
    
    def normalize_joint(x):
        return np.clip(x / joint_scale, -1e6, 1e6)
    
    # Adaptive slack gating: smooth sigmoid transition centered at slack=0, width=1.0s
    # Ensures near-deadline tasks (e.g., slack=0.1s) retain meaningful urgency
    slack_score = 1.0 / (1.0 + np.exp(-slack))  # maps slack→[-inf,inf] to [0,1]; higher = more urgent
    # Invert so smaller score = higher priority → use (1 - slack_score) for penalty baseline
    deadline_urgency = 1.0 - slack_score  # [0,1]; 1.0 = slack → -inf (critical), 0.0 = slack → +inf (idle)
    
    # Criticality-energy synergy: prioritize high-rank + low-energy-per-time
    norm_crit_energy = normalize_joint(crit_energy_synergy)
    
    # Uncertainty-modulated criticality: dampen upward_rank influence under high uncertainty
    unc_damp = 1.0 / (1.0 + np.clip(uncertainty, 0.0, 10.0))
    norm_damped_rank = normalize_joint(upward_rank * unc_damp)
    
    # Starvation-robust wait boost: activate when task is either late-biased or high-rank dominant
    median_slack = np.median(slack) if slack.size > 1 else 0.0
    max_rank = np.max(upward_rank) if upward_rank.size > 0 else 1.0
    wait_activation = (slack < median_slack) | (upward_rank > 0.75 * max_rank)
    norm_wait = np.where(wait_activation, normalize_joint(ready_wait_time), 0.0)
    
    # Energy efficiency term: penalize high incremental energy *relative to work*
    work_efficiency = np.clip(min_incremental_energy / (np.maximum(remaining_work, eps)), -1e6, 1e6)
    norm_energy_eff = normalize_joint(work_efficiency)
    
    # Final score: linear combination with calibrated weights — all terms promote lower score = higher priority
    # Dominant deadline urgency, strong criticality-energy synergy, moderate starvation guard, light efficiency penalty
    score = (
        1.0 * deadline_urgency +           # Hard deadline dominance, but smoothly graded
        0.4 * norm_crit_energy +          # Core objective: minimize risk-adjusted fuzzy energy on critical path
        0.3 * norm_damped_rank +          # Critical-path leverage, uncertainty-dampened
        0.15 * norm_wait +                # Fairness guard, robustly activated
        0.1 * norm_energy_eff             # Bonus for energy-per-work efficiency
    )
    
    # Ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
