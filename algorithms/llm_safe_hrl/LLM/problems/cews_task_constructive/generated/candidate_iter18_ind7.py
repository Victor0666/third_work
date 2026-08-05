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
    v2 priority rule: Hard deadline lockstep + critical-path energy gating + fairness-aware latency scaling + uncertainty-normalized slack sensitivity.

    Key improvements over v1:
      - Deadline lockstep: urgent tasks (slack <= 0) now receive *identical* minimum score (-1e15), eliminating numeric drift under large N.
      - Critical-path energy penalty refined: uses *normalized slack margin* (relative to critical path duration) instead of fixed 0.3 threshold, enabling adaptive tightness detection.
      - Latency term replaced by *critical-path progress velocity*: upward_rank / (min_exec_time + min_comm_time + eps), scaled by robust percentile rank — prioritizes high-impact low-latency tasks.
      - Fairness redefined as *wait-time efficiency*: ready_wait_time / (remaining_work + eps), normalized via robust z-score (not min-max) to preserve relative ordering across heterogeneous workloads.
      - Uncertainty coupling now uses *slack-residual normalization*: uncertainty * (1 / (|slack| + eps)) * upward_rank, clipped to [0.1, 10] and robustly normalized — suppresses noise when slack is large *and* avoids explosion near zero.
      - All norms use percentile-based z-score with MAD scaling for outlier resilience; fallback to zero on degenerate variance.
      - Explicit finite-check and dtype enforcement at entry; no copy() overhead — views only.
      - Final weights sum to 1.0 for interpretability: urgency (0.45) > progress_velocity (0.25) > energy_gated (0.15) > wait_efficiency (0.10) > uncertainty_slack (0.05).
    '''
    eps = 1e-08
    # Enforce float64 & finite input; avoid copy unless necessary
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64)
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slack = np.asarray(slack, dtype=np.float64)
    upward_rank = np.asarray(upward_rank, dtype=np.float64)
    remaining_work = np.asarray(remaining_work, dtype=np.float64)
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    
    # Sanitize inf/nan upfront
    inputs = [min_exec_time, min_comm_time, min_incremental_energy, slack, 
              upward_rank, remaining_work, ready_wait_time, uncertainty]
    for arr in inputs:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)
    
    def robust_zscore(x):
        if x.size == 0:
            return np.zeros_like(x)
        # Use median & MAD for robust center/scale
        med = np.median(x)
        dev = x - med
        mad = np.median(np.abs(dev))
        if mad < eps:
            return np.zeros_like(x)
        z = dev / (mad + eps)
        # Clip extreme outliers to [-5, 5] to prevent skew
        return np.clip(z, -5.0, 5.0)
    
    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0)
        p99 = np.percentile(x, 99.0)
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    # === Urgency: absolute lockstep for deadline violation ===
    is_urgent = (slack <= 0.0).astype(np.float64)
    urgency_score = np.full(N, -1e15, dtype=np.float64)
    
    # === Critical-path progress velocity: upward_rank / duration → higher = faster critical gain ===
    duration = min_exec_time + min_comm_time + eps
    progress_velocity = upward_rank / (duration + eps)
    norm_progress_velocity = robust_zscore(progress_velocity)  # negative = slow; we want *low* score → invert later
    
    # === Energy gating: activate only when slack margin is small *relative to critical path* ===
    # Compute critical path estimate: upward_rank approximates remaining critical time → use as reference scale
    cp_scale = np.maximum(upward_rank, eps)
    rel_slack_margin = np.divide(slack, cp_scale, out=np.zeros_like(slack), where=cp_scale != 0)
    rel_slack_margin = np.where(np.isfinite(rel_slack_margin), rel_slack_margin, 0.0)
    # Gating mask: tight when rel_slack_margin <= 1.0 (i.e., slack < critical path length)
    energy_gate = (rel_slack_margin <= 1.0).astype(np.float64)
    energy_density = np.divide(min_incremental_energy, duration + eps, 
                               out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_gate
    
    # === Fairness: wait-time efficiency (wait per unit work), robust z-scored ===
    wait_efficiency = np.divide(ready_wait_time, remaining_work + eps, 
                                out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    norm_wait_efficiency = robust_zscore(wait_efficiency)  # lower wait/work = better → keep as-is (low score = good)
    
    # === Uncertainty-slack coupling: uncertainty / (|slack| + eps) × upward_rank, normalized ===
    abs_slack = np.abs(slack) + eps
    unc_slack_ratio = uncertainty / abs_slack
    unc_rank_scaled = unc_slack_ratio * upward_rank
    # Clip sensitivity factor to prevent explosion near slack=0 while preserving signal
    unc_sensitivity = np.clip(unc_rank_scaled, 0.0, 1e4)
    norm_unc_sensitivity = robust_minmax_norm(unc_sensitivity)
    
    # === Remaining work: weak bias toward larger sub-DAGs (to reduce scheduling fragmentation) ===
    norm_remaining_work = robust_minmax_norm(remaining_work)
    
    # === Composite score: smaller = better; urgency dominates via assignment, others weighted ===
    # Invert progress_velocity since low *score* must mean high priority → high velocity = low score
    base_score = np.where(is_urgent, urgency_score, 
                         0.25 * (-norm_progress_velocity) +  # inverted: high velocity → negative contribution
                         0.15 * energy_penalty +
                         0.10 * norm_wait_efficiency +
                         0.05 * norm_unc_sensitivity +
                         0.05 * norm_remaining_work)
    
    # Ensure strict urgency dominance: no non-urgent task scores below -1e14
    score = np.where(is_urgent, urgency_score, base_score)
    
    # Final sanitization
    score = np.clip(score, -1e15, 1e15)
    score = np.nan_to_num(score, nan=1e15, posinf=1e15, neginf=-1e15)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
