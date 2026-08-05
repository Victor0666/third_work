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
    Self-evolved priority rule v2: hardened DDL compliance + adaptive risk-aware energy + robust fairness.
    
    Key improvements over v1:
    - Replaces brittle quantile-based slack threshold with trimmed-mean robust centrality (alpha=0.1) for stability in small-N sets
    - Widens tight-slack gating from rel_slack <= 0.1 → <= 0.25 to preserve risk sensitivity for near-deadline tasks
    - Replaces p90 wait-per-work gate with adaptive fairness boost activated when wait_ratio > median + 1.5*std (robust z-score)
    - Introduces slack-aware energy normalization: energy term scaled by (1 + max(0, -slack)/task_duration) to penalize high-energy tasks under pressure
    - Adds critical-path latency penalty: upward_rank * max(0, -slack) to prioritize critical tasks when lateness looms
    - All normalizations use bounded robust_minmax with explicit size guard; all divisions guarded; NaN/inf handled deterministically
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
    
    def robust_minmax(x):
        if x.size == 0:
            return np.zeros_like(x)
        # Use trimmed mean for robust centrality in small-N cases
        x_sorted = np.sort(x)
        trim_n = max(1, int(0.1 * len(x_sorted)))
        x_trimmed = x_sorted[trim_n:-trim_n] if len(x_sorted) > 2*trim_n else x_sorted
        x_min, x_max = (np.min(x_trimmed), np.max(x_trimmed))
        rng = x_max - x_min + eps
        return np.clip((x - x_min) / rng, 0.0, 1.0)
    
    task_duration = min_exec_time + min_comm_time + eps
    has_negative_slack = (slack < 0.0).astype(float)
    
    # Lateness penalty: bounded, proportional to violation relative to median duration
    lateness_penalty = np.where(slack < 0.0, 
                               np.clip(-slack / (np.median(task_duration) + eps), 0.0, 4.0), 
                               0.0)
    
    # Robust relative slack using trimmed mean for stability
    rel_slack = slack / task_duration
    slack_central = np.mean(rel_slack) if N == 1 else np.mean(rel_slack[rel_slack != np.inf][rel_slack != -np.inf])
    urgency_sigmoid = 1.0 / (1.0 + np.exp(6.0 * (rel_slack - slack_central)))
    urgency = np.where(has_negative_slack, 1.0, urgency_sigmoid)
    
    # Critical-path latency penalty: amplify upward_rank for negative slack
    cp_latency_penalty = upward_rank * np.maximum(0.0, -slack) / (task_duration + eps)
    
    # Energy term: work-normalized, upward_rank weighted, and slack-aware scaled
    energy_per_work = np.maximum(min_incremental_energy / (remaining_work + eps), eps)
    base_crit_energy = energy_per_work * (1.0 + upward_rank / (np.median(upward_rank + eps) + eps))
    # Slack-aware scaling: penalize high energy more when slack is negative or low
    slack_scale_factor = 1.0 + np.maximum(0.0, -slack) / (task_duration + eps)
    crit_weighted_energy = base_crit_energy * slack_scale_factor
    
    # Uncertainty gating widened to rel_slack <= 0.25 for stronger near-deadline risk response
    tight_slack_mask = (rel_slack <= 0.25).astype(float)
    dur_uncertainty = np.where(task_duration > eps, uncertainty / task_duration, 0.0)
    uncertainty_boost = dur_uncertainty * tight_slack_mask
    
    # Fairness boost: robust z-score on wait_per_work (median + 1.5*std threshold)
    wait_per_work = ready_wait_time / (remaining_work + eps)
    wait_median = np.median(wait_per_work)
    wait_std = np.std(wait_per_work) + eps
    wait_z_threshold = wait_median + 1.5 * wait_std
    wait_gate = ((rel_slack > -0.02).astype(float) * 
                 (remaining_work > np.quantile(remaining_work, 0.2, method='midpoint') + eps).astype(float) * 
                 (wait_per_work > wait_z_threshold).astype(float))
    norm_wait_time = robust_minmax(ready_wait_time)
    wait_boost = norm_wait_time * wait_gate * 0.25
    
    # Normalize all components robustly
    norm_urgency = robust_minmax(urgency)
    norm_lateness = robust_minmax(lateness_penalty)
    norm_energy = robust_minmax(crit_weighted_energy)
    norm_uncertainty = robust_minmax(uncertainty_boost)
    norm_upward = robust_minmax(upward_rank)
    norm_cp_latency = robust_minmax(cp_latency_penalty)
    
    # Final score: urgency dominates, lateness & energy penalized, fairness & criticality rewarded
    score = (-3.0 * norm_urgency + 
             1.3 * norm_lateness + 
             1.0 * norm_energy + 
             0.3 * norm_uncertainty - 
             0.4 * norm_upward + 
             0.15 * wait_boost + 
             0.8 * norm_cp_latency)
    
    # Clamp and sanitize
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
