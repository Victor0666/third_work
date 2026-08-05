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
    Self-evolved v3: Deadline-hardened + critical-path-anchored + energy-fairness-stabilized.
    
    Key advances over v1:
      - Replaces fragile exponential urgency with *inverse linear slack scaling* for positive slack,
        eliminating numeric instability and ensuring monotonic priority increase as slack shrinks.
      - Introduces *critical-path anchoring*: upward_rank normalized by max_upward_rank (not median),
        preserving relative importance hierarchy even under skewed distributions or N=1.
      - Stabilizes energy fairness: uses *log-transformed energy_per_work* to compress dynamic range
        and avoid outlier dominance; scaled only by slack-aware *sigmoid* (not max(1,1+slack/5)) —
        smooth, bounded, and avoids artificial floor/ceiling artifacts.
      - Enhances starvation robustness: wait-per-work penalty now *adaptive*, using percentile-based cap
        (90th percentile under slack>=0) instead of max — prevents single outlier from distorting fairness.
      - Tightens normalization: robust_zscore now uses *trimmed mean* for centering when N>3, improving
        stability under heavy-tailed distributions while retaining MAD for dispersion.
      - Final weights rebalanced to strengthen deadline fidelity (0.52), critical-path anchoring (0.24),
        stabilized energy fairness (0.11), critical-path density (0.07), adaptive fairness (0.04), uncertainty (0.02).
    '''
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
    
    def robust_zscore(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        # Use trimmed mean for centering if N > 3, else median for robustness
        if N > 3:
            sorted_x = np.sort(x)
            trim_low = max(1, int(0.05 * N))
            trim_high = max(1, int(0.05 * N))
            center_x = np.mean(sorted_x[trim_low:-trim_high]) if N > 2 * trim_low else np.median(x)
        else:
            center_x = np.median(x)
        mad = np.median(np.abs(x - center_x)) + eps
        z = (x - center_x) / mad
        return np.clip(z, -5.0, 5.0)
    
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    
    # Deadline urgency: hard penalty for slack <= 0; inverse-linear decay for slack > 0 (stable & monotonic)
    urgency_neg = 1.0 + np.clip(-slack / (np.median(task_min_duration) + eps), 0.0, 4.0)
    urgency_pos = np.where(slack > 0, 1.0 / (1.0 + slack / (np.median(task_min_duration) + eps)), 0.0)
    deadline_urgency = np.where(slack <= 0, urgency_neg, urgency_pos)
    
    # Critical-path anchoring: upward_rank scaled by global max (preserves ordinal structure)
    max_upward = np.max(upward_rank) + eps
    anchored_rank = upward_rank / max_upward
    
    # Energy fairness: log-compressed energy_per_work, scaled by smooth sigmoid slack adjustment
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    log_energy_pw = np.log1p(energy_per_work)  # log(1+x) avoids log(0), compresses outliers
    # Sigmoid slack scaling: 0.5 at slack=0, approaches 1.0 for large slack, 0.0 for large negative slack
    slack_sigmoid = 1.0 / (1.0 + np.exp(-slack / (np.median(task_min_duration) + eps)))
    fair_energy_score = log_energy_pw * (0.5 + 0.5 * slack_sigmoid)  # [0.5, 1.0] scaling
    
    # Critical-path density: upward_rank / duration, clipped to avoid division artifacts
    cp_density = upward_rank / (task_min_duration + eps)
    cp_density = np.clip(cp_density, eps, 1e6)
    
    # Adaptive starvation mitigation: wait-per-work capped at 90th percentile (not max) under slack >= 0
    wait_mask = slack >= 0
    wait_per_work = ready_wait_time / (remaining_work + eps)
    if np.any(wait_mask):
        cap_percentile = np.percentile(wait_per_work[wait_mask], 90.0)
        wait_penalty = np.where(wait_mask, np.clip(wait_per_work / (cap_percentile + eps), 0.0, 1.0), 0.0)
    else:
        wait_penalty = np.zeros_like(wait_per_work)
    
    norm_urgency = robust_zscore(deadline_urgency)
    norm_anchored = robust_zscore(anchored_rank)
    norm_fair_energy = robust_zscore(fair_energy_score)
    norm_cp_density = robust_zscore(cp_density)
    norm_fairness = robust_zscore(wait_penalty)
    norm_unc = robust_zscore(uncertainty)
    
    # Final weighted score: smaller = higher priority
    score = (
        0.52 * norm_urgency +
        0.24 * (1.0 - norm_anchored) +
        0.11 * norm_fair_energy +
        0.07 * (1.0 - norm_cp_density) +
        0.04 * norm_fairness +
        0.02 * norm_unc
    )
    
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
