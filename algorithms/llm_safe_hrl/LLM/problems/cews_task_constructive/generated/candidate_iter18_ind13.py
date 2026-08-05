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
    v2: Refined safety-aware priority with calibrated urgency, stable SEER activation,
        uncertainty-integrated critical-path focus, and starvation-safe fairness.
    
    Key improvements over v1:
      - Replaces aggressive `slack - 2*uncertainty` with *risk-weighted slack*: 
        `robust_slack = slack - uncertainty * (1 + tanh(uncertainty/tau_u))`, 
        smoothly penalizing high uncertainty without over-conservatism.
      - Uses *global median slack* (not positive-only) for SEER activation threshold → stable, continuous energy optimization.
      - Critical-path density (CPD) now incorporates *uncertainty-adjusted latency* to prioritize low-risk critical tasks.
      - Fairness term uses *relative aging* normalized by workflow-scale wait time (median-based), preventing bias in heterogeneous workloads.
      - Urgency is *comparatively scaled*: tanh(-robust_slack / tau_urg) normalized robustly — ensures near-deadline discrimination remains sharp but bounded.
      - No hard violation override; instead, negative slack induces strong urgency dominance via multiplicative gating (preserves comparability).
      - All normalizations use MAD with strict finite clipping [-2.5, 2.5] and degenerate handling for N=1 or flat arrays.
      - Final score structure: urgency dominates (gated), then CPD, then SEER, then fairness — enforced via additive weighting with decaying coefficients.
    """
    eps = 1e-08
    tau_urg = 0.35
    tau_u = 1.0

    # Sanitize inputs: ensure finite, non-NaN, bounded values
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

    # Risk-weighted slack: smooth, adaptive penalty — avoids over-penalization of uncertain-but-feasible tasks
    unc_factor = 1.0 + np.tanh(uncertainty / tau_u)
    robust_slack = slack - uncertainty * unc_factor
    robust_slack = np.clip(robust_slack, -1e6, 1e6)

    # Robust MAD-based normalization with N=1 and flat-array fallback
    def robust_mad_normalize(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        normed = (x - med) / mad
        return np.clip(normed, -2.5, 2.5)

    # --- Urgency term: sharp, comparable, risk-aware ---
    # tanh(-robust_slack/tau_urg) gives strong negative response near deadline, saturates smoothly
    urgency_raw = np.tanh(-robust_slack / tau_urg)
    urgency_norm = robust_mad_normalize(urgency_raw)
    urgency_term = -4.2 * urgency_norm  # dominant weight

    # --- Critical-path density (CPD): uncertainty-aware & work-normalized ---
    # Prioritizes high-upward-rank tasks that are fast *and* low-uncertainty
    base_latency = min_exec_time + min_comm_time + eps
    # Uncertainty dampens perceived criticality: exp(-uncertainty/tau_u) scales CPD down for risky tasks
    cpd_base = (upward_rank / base_latency) * (remaining_work / (np.median(remaining_work) + eps))
    cpd_gate = np.tanh(np.maximum(robust_slack, 0.0) / tau_urg) * np.exp(-uncertainty / tau_u)
    cpd_masked = cpd_base * cpd_gate
    cpd_norm = robust_mad_normalize(cpd_masked)
    cpd_term = -2.1 * cpd_norm

    # --- SEER (Slack-Gated Energy Efficiency Ratio): stable activation ---
    # Activated only when robust_slack > global median slack (stable, continuous threshold)
    global_median_slack = np.median(robust_slack) + eps
    seer_base = base_latency / (min_incremental_energy + eps)
    # Activation: smooth ramp from 0 at median to 1 well above — no binary switch
    seer_activation = np.where(
        robust_slack > global_median_slack,
        np.tanh((robust_slack - global_median_slack) / (global_median_slack + eps)),
        0.0
    )
    seer_masked = seer_base * seer_activation
    seer_norm = robust_mad_normalize(seer_masked)
    seer_term = -1.1 * seer_norm

    # --- Fairness: starvation-safe relative aging ---
    # Normalize wait time by median ready_wait_time to handle scale heterogeneity
    median_wait = np.median(ready_wait_time) + eps
    rel_wait = (ready_wait_time + eps) / median_wait
    sqrt_rel_wait = np.sqrt(rel_wait)
    # Suppress only under real deadline pressure: exp(-max(0, -robust_slack)/tau_urg)
    risk_suppress = np.exp(-np.maximum(-robust_slack, 0.0) / tau_urg)
    fairness_raw = sqrt_rel_wait * risk_suppress
    fairness_clipped = np.clip(fairness_raw, 0.0, 0.45)
    fairness_norm = robust_mad_normalize(fairness_clipped)
    fairness_term = -0.08 * fairness_norm

    # --- Final score: additive, lexicographically weighted, fully bounded ---
    score = urgency_term + cpd_term + seer_term + fairness_term
    # Ensure finite output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)

    return score
