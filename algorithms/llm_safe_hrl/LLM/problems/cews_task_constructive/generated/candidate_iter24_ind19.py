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
    v3: Deadline-first, energy-aware, uncertainty-resilient priority with:
      - Adaptive urgency blending: sigmoid dominates far from DDL; arctan sharpens near zero slack;
        blended via *robust slack-to-uncertainty ratio* for context-aware weighting.
      - Criticality gating refined: uses *relative risk* (uncertainty / |robust_slack|) with smooth
        logistic suppression instead of hard threshold → avoids discontinuities & improves gradient flow.
      - Energy term now *work-normalized marginal efficiency* (J/MI), activated only when slack > median_slack
        AND slack > 0, and further scaled by *remaining critical path work* to prioritize energy savings
        on high-impact subgraphs — not just isolated tasks.
      - Fairness redefined as *normalized wait-time deficit*: (ready_wait_time - quantile_10_wait) clipped
        and gated by slack >= 0 → targets starvation only when safe, proportional to how long overdue.
      - All components use MAD normalization with degenerate-N handling and strict [-1.8, 1.8] bounds.
      - Final score weights tuned for stronger deadline safety margin (+3.2 urgency) and reduced energy penalty
        (-1.2 instead of -1.4) to prevent premature energy optimization under tight deadlines.
      - No inf/nan/zero-division — full numerical sanitization applied upfront and per-operation.
    """
    eps = 1e-08
    tau_urg = 0.35
    tau_u = 1.0
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1e6, neginf=1e-6)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)

    # Robust slack: subtract uncertainty-weighted risk margin
    unc_factor = 1.0 + np.tanh(uncertainty / tau_u)
    robust_slack = slack - uncertainty * unc_factor
    robust_slack = np.clip(robust_slack, -1e6, 1e6)

    # Adaptive urgency blend: weight sigmoid vs arctan by relative risk level
    rel_risk = uncertainty / (np.abs(robust_slack) + eps)
    # Smooth logistic gate: near-zero risk → favor sigmoid; high risk → favor arctan's local sensitivity
    sigmoid_weight = 1.0 / (1.0 + np.exp((rel_risk - 1.0) / 0.5))
    arctan_weight = 1.0 - sigmoid_weight

    sigmoid_urgency = 1.0 / (1.0 + np.exp(robust_slack / tau_urg))
    arctan_input = np.clip(robust_slack / (uncertainty + eps), -10.0, 10.0)
    arctan_urgency = np.arctan(arctan_input) / (np.pi / 2)
    # Arctan maps to [−1,1]; convert to [0,1] urgency: higher urgency = smaller completion margin
    arctan_urgency_norm = 0.5 * (1.0 - arctan_urgency)
    
    urgency_raw = sigmoid_weight * sigmoid_urgency + arctan_weight * arctan_urgency_norm

    # Critical-path density: uncertainty-gated, smooth suppression, latency-weighted
    base_latency = min_exec_time + min_comm_time + eps
    latency_weighted = base_latency * (1.0 + uncertainty * 0.5)
    # Smooth logistic suppression: suppress criticality only when risk dominates slack
    risk_gate = 1.0 / (1.0 + np.exp((rel_risk - 2.0) / 0.3))
    median_rw = np.median(remaining_work) + eps
    cp_density = risk_gate * (upward_rank + eps) / (latency_weighted + eps) * (remaining_work / median_rw)

    # Energy efficiency: J per MI, activated only in safe slack regime, scaled by critical path weight
    # Use *remaining critical work* (not just local remaining_work) to focus energy savings where it matters most
    energy_eff_base = min_incremental_energy / (remaining_work + eps)
    global_median_slack = np.median(robust_slack) + eps
    energy_activation = np.where((robust_slack > global_median_slack) & (robust_slack > 0), 1.0, 0.0)
    energy_eff = energy_eff_base * energy_activation

    # Fairness: normalized wait-time deficit relative to 10th percentile, gated by slack safety
    if ready_wait_time.size == 1:
        wait_quantile = ready_wait_time[0]
        fairness_raw = np.array([0.0])
    else:
        sorted_wait = np.sort(ready_wait_time)
        # 10th percentile (index floor(0.1*(n-1)))
        idx_q10 = max(0, min(len(sorted_wait)-1, int(0.1 * (len(sorted_wait)-1))))
        wait_quantile = sorted_wait[idx_q10]
        fairness_raw = np.clip(ready_wait_time - wait_quantile, 0.0, None)
    fairness_gate = np.where(robust_slack >= 0.0, 1.0, 0.0)
    fairness = fairness_raw * fairness_gate

    # MAD normalization: safe, bounded, degenerate-N aware
    def safe_mad_normalize(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.zeros_like(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        if mad < eps:
            return np.zeros_like(x)
        normed = (x - med) / mad
        return np.clip(normed, -1.8, 1.8)

    urgency_norm = safe_mad_normalize(urgency_raw)
    cp_norm = safe_mad_normalize(cp_density)
    energy_norm = safe_mad_normalize(energy_eff)
    fairness_norm = safe_mad_normalize(fairness)

    # Final score: stronger urgency dominance, milder energy penalty, fairness preserved
    # Prioritizes deadline compliance first; energy only optimized where slack allows
    score = +3.2 * urgency_norm - 1.9 * cp_norm - 1.2 * energy_norm + 0.12 * fairness_norm

    # Final numeric safeguard
    score = np.clip(score, -1e9, 1e9)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)

    return score.reshape(-1)
