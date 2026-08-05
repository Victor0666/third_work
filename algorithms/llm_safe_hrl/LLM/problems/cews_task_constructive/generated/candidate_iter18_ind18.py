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
    v2 evolution: Fixes urgency under-prioritization of critically late tasks by using tail-aware tau;
                   restores correct SEER soft-gating polarity (penalize energy *only* when slack is tight);
                   replaces per-term rank normalization with *global percentile-based scaling* for consistent term comparability;
                   introduces *latency-criticality coupling*: multiplies urgency and CP-density to jointly prioritize
                   high-importance + high-risk tasks; adds *starvation-robust fairness* via wait-time quantile gating;
                   eliminates fragile percentile thresholds in favor of adaptive, distribution-anchored clipping.
    '''
    eps = 1e-08
    def clean(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=eps)

    min_exec_time = clean(min_exec_time)
    min_comm_time = clean(min_comm_time)
    min_incremental_energy = clean(min_incremental_energy)
    slack = clean(slack)
    upward_rank = clean(upward_rank)
    remaining_work = clean(remaining_work)
    ready_wait_time = clean(ready_wait_time)
    uncertainty = clean(uncertainty)

    # Global normalization: map each term to [0,1] via percentile-based scaling (stable for N=1)
    def global_percentile_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        if x.size == 1:
            return np.array([0.5])
        # Use 5th and 95th percentiles for robust range, avoid outliers
        p5 = np.percentile(x, 5)
        p95 = np.percentile(x, 95)
        denom = p95 - p5 + eps
        normed = (x - p5) / denom
        return np.clip(normed, 0.0, 1.0)

    # Urgency: adaptive sigmoid with tail-aware steepness (tau based on slack 10th percentile)
    # Critical tasks (low slack) get sharp penalty; τ shrinks as tail risk increases
    slack_p10 = np.percentile(slack, 10)
    tau_urgency = np.clip(0.5 + 1.0 * np.maximum(0.0, -slack_p10), 0.3, 4.0)
    # Raw urgency: high when slack is negative/low → map to [0,1] where 1 = most urgent
    urgency_raw = 1.0 / (1.0 + np.exp((slack + 1.0) / tau_urgency))
    norm_urgency = global_percentile_norm(urgency_raw)
    urgency_term = -7.0 * norm_urgency  # Strongest weight: deadline hard constraint first

    # SEER (Slack-Conditioned Energy Efficiency Ratio): only penalize energy when slack is tight
    exec_comm_sum = min_exec_time + min_comm_time + eps
    base_seer = min_incremental_energy / exec_comm_sum
    # Soft gate: only activate energy penalty when slack <= median (i.e., risk zone)
    slack_gate = 1.0 / (1.0 + np.exp((slack - np.median(slack)) / 2.0))
    seer_soft_gated = base_seer * slack_gate
    norm_seer = global_percentile_norm(seer_soft_gated)
    seer_term = -1.5 * norm_seer  # Energy minimization secondary, but active only under pressure

    # Critical-path density: upward_rank * work / latency → importance per time unit
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    # Coupling: multiply with urgency to jointly prioritize critical *and* urgent paths
    cp_coupled = cp_density * (1.0 + urgency_raw)  # Boosts high-upward-rank tasks that are also urgent
    norm_cp_coupled = global_percentile_norm(cp_coupled)
    cp_density_term = 1.2 * norm_cp_coupled

    # Fairness: prevent starvation via wait-time quantile gating — only boost long-waiting tasks *if* slack permits
    # Avoid boosting waiting tasks that are already deadline-critical (they’re covered by urgency)
    wait_safe_mask = slack > np.percentile(slack, 25)  # Only apply fairness when slack allows
    rel_wait = np.where(wait_safe_mask, ready_wait_time, 0.0)
    norm_wait = global_percentile_norm(rel_wait)
    # Log-scaled boost ensures strict ordering and bounded effect
    fairness_boost = np.log1p(norm_wait * 100.0) / np.log1p(100.0)
    fairness_term = -0.3 * global_percentile_norm(fairness_boost)

    # Uncertainty modulation: only penalize communication when both slack > 0 AND uncertainty is extreme
    unc_p90 = np.percentile(uncertainty, 90)
    high_unc_mask = (uncertainty > unc_p90) & (slack > 0)
    comm_unc_penalty = np.where(high_unc_mask, min_comm_time * uncertainty * 0.3, 0.0)
    # Normalize penalty globally to avoid dominance
    norm_unc_penalty = global_percentile_norm(comm_unc_penalty)
    unc_term = 0.1 * norm_unc_penalty

    # Assemble final score: smaller = better
    score = urgency_term + seer_term + cp_density_term + fairness_term + unc_term

    # Final sanitization: ensure finite, deterministic, shape-(N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)

    return score
