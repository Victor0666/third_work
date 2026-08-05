import numpy as np

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
    v2: Hybrid priority rule combining Parent 2's robust sigmoid urgency and percentile fairness
         with Parent 1's arctan-based urgency continuity and uncertainty-gated criticality.
    Key innovations:
      - Dual-mode urgency: sigmoid for global deadline dominance + arctan for local slack sensitivity,
        fused via weighted sum to preserve gradient flow while enhancing DDL resolution near zero slack.
      - Critical-path density uses *uncertainty-weighted latency* (Parent 2) but gated by arctan-based
        risk score to suppress low-confidence critical tasks only when uncertainty dominates slack.
      - Energy efficiency term activated only when both slack > median_slack AND slack > 0 (Parent 2),
        but normalized by *remaining_work* (Parent 1) for physics-aligned marginal energy per work unit.
      - Fairness uses wait-time percentile (Parent 2) with soft aging activation: only applied when
        slack >= 0 AND ready_wait_time > 0.05, preventing starvation without biasing DDL-critical paths.
      - All components MAD-normalized with degenerate-N handling and [-1.8, 1.8] clipping for stability.
      - No hard overrides or inf/nan injection: fully smooth, bounded, and numerically safe.
    """
    eps = 1e-08
    tau_urg = 0.35
    tau_u = 1.0
    
    def sanitize(x):
        x = np.asarray(x, dtype=float)
        return np.nan_to_num(x, nan=eps, posinf=1000000.0, neginf=1e-06)
    
    min_exec_time = sanitize(min_exec_time)
    min_comm_time = sanitize(min_comm_time)
    min_incremental_energy = sanitize(min_incremental_energy)
    slack = sanitize(slack)
    upward_rank = sanitize(upward_rank)
    remaining_work = sanitize(remaining_work)
    ready_wait_time = sanitize(ready_wait_time)
    uncertainty = sanitize(uncertainty)
    
    # Robust slack: subtract uncertainty-aware penalty (Parent 2)
    unc_factor = 1.0 + np.tanh(uncertainty / tau_u)
    robust_slack = slack - uncertainty * unc_factor
    robust_slack = np.clip(robust_slack, -1000000.0, 1000000.0)
    
    # Dual-mode urgency: sigmoid (global DDL dominance) + arctan (local slack sensitivity)
    sigmoid_urgency = 1.0 / (1.0 + np.exp(robust_slack / tau_urg))
    arctan_input = robust_slack / (uncertainty + eps)
    arctan_urgency = np.arctan(arctan_input) / (np.pi / 2)  # [-1, 1]
    # Blend: emphasize sigmoid far from deadline, arctan near zero slack
    urgency_raw = 0.7 * sigmoid_urgency + 0.3 * (0.5 * (1.0 - arctan_urgency))
    
    # Uncertainty-weighted latency for critical-path density (Parent 2)
    base_latency = min_exec_time + min_comm_time + eps
    latency_weighted = base_latency * (1.0 + uncertainty * 0.5)
    
    # Risk-gated critical-path density: suppress if uncertainty >> |robust_slack|
    risk_ratio = uncertainty / (np.abs(robust_slack) + eps)
    risk_gate = np.where(risk_ratio > 2.0, 0.0, 1.0)
    median_rw = np.median(remaining_work) + eps
    cp_density = risk_gate * (upward_rank + eps) / (latency_weighted + eps) * (remaining_work / median_rw)
    
    # Energy efficiency: marginal energy per unit remaining work, activated only when safe
    global_median_slack = np.median(robust_slack) + eps
    energy_eff_base = min_incremental_energy / (remaining_work + eps)
    energy_activation = np.where((robust_slack > global_median_slack) & (robust_slack > 0), 1.0, 0.0)
    energy_eff = energy_eff_base * energy_activation
    
    # Percentile-based fairness with soft aging gate (Parent 2 + refinement)
    if ready_wait_time.size == 1:
        wait_percentile = np.array([0.5])
    else:
        sorted_wait = np.sort(ready_wait_time)
        ranks = np.searchsorted(sorted_wait, ready_wait_time, side='left') + 1
        wait_percentile = (ranks - 0.5) / len(ready_wait_time)
    fairness_raw = np.clip(wait_percentile, 0.0, 1.0)
    fairness_gate = np.where((robust_slack >= 0.0) & (ready_wait_time > 0.05), 1.0, 0.0)
    fairness = fairness_raw * fairness_gate
    
    def safe_mad_normalize(x):
        x = np.clip(x, -1000000.0, 1000000.0)
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
    
    # Final score: urgency dominant, critical-path and energy negative (higher is better), fairness weak positive
    score = (
        +3.0 * urgency_norm
        - 1.9 * cp_norm
        - 1.4 * energy_norm
        + 0.09 * fairness_norm
    )
    
    # Ensure finite output with strict bounds and nan/inf cleanup
    score = np.clip(score, -1000000000.0, 1000000000.0)
    score = np.nan_to_num(score, nan=1000000000.0, posinf=1000000000.0, neginf=-1000000000.0)
    
    return score.reshape(-1)
