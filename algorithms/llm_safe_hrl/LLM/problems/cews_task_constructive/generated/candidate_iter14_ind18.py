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
    v2 hybrid: Combines Parent 2's lexicographic urgency dominance and slack-gated SEER
    with Parent 1's robust MAD normalization and edge-aware communication risk modulation.
    Key innovations:
    - Urgency: tanh-based risk saturation (Parent 2) + linear slack penalty for violation cases (Parent 1 fallback)
    - Energy-efficiency: SEER = (energy / (exec+comm)) * exp(-max(0,slack)/tau) — only active when slack >= 0
    - Criticality-latency coupling: uses upward_rank * remaining_work / (exec+comm+eps) normalized robustly
    - Edge-risk comm penalty: applied only when min_incremental_energy < median_energy AND slack > 0
    - Fairness: linear wait boost gated by slack > tau_safe, no tanh distortion → cleaner signal
    - Uncertainty boost: only active for high uncertainty AND positive slack, clipped to avoid noise
    - All terms normalized via MAD with epsilon-robust fallback; final weights enforce strict urgency ≫ SEER ≫ CP ≫ fairness ≫ uncertainty
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

    def normalize_robust(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad > eps:
            scale = mad + eps
        else:
            xmin, xmax = np.min(x), np.max(x)
            scale = max(xmax - xmin, eps)
        return (x - med) / (scale + eps)

    # === URGENCY TERM (dominant, negative score = higher priority) ===
    # tanh saturation for risk scaling + linear penalty when slack < 0 to force immediate action
    urgency_tanh = np.tanh(-slack / 2.0)
    urgency_linear = np.where(slack < 0, -slack * 0.5, 0.0)
    urgency_raw = urgency_tanh + urgency_linear
    norm_urgency = normalize_robust(urgency_raw)
    urgency_term = -4.0 * norm_urgency  # strongest weight

    # === SLACK-GATED SEER TERM (energy efficiency under deadline headroom) ===
    exec_comm_sum = min_exec_time + min_comm_time + eps
    base_seer = min_incremental_energy / exec_comm_sum
    seer_slack_factor = np.exp(-np.maximum(0.0, slack) / 5.0)
    seer_masked = np.where(slack >= 0, base_seer * seer_slack_factor, 0.0)
    norm_seer = normalize_robust(seer_masked)
    seer_term = -1.6 * norm_seer  # second strongest, promotes energy savings only when safe

    # === CRITICAL-PATH DELAY DENSITY (CPDD): importance per latency unit) ===
    cpdd = upward_rank * remaining_work / (exec_comm_sum + eps)
    # Modulate with edge-risk-comm penalty only when energy-efficient (edge-bound) AND slack permits
    median_energy = np.median(min_incremental_energy) + eps
    edge_risk_mask = (min_incremental_energy < median_energy).astype(float)
    comm_risk_boost = min_comm_time * uncertainty * edge_risk_mask * 0.3
    cpdd_penalized = cpdd + np.where(slack > 0, comm_risk_boost / (exec_comm_sum + eps), 0.0)
    norm_cpdd = normalize_robust(cpdd_penalized)
    cpdd_term = 1.2 * norm_cpdd  # positive: higher density → lower priority unless urgency overrides

    # === FAIRNESS TERM (linear wait boost, only when slack > safe threshold) ===
    wait_boost = np.where(slack > 3.0, np.clip(ready_wait_time * 0.1, 0.0, 0.3), 0.0)
    norm_wait = normalize_robust(wait_boost)
    wait_term = -0.2 * norm_wait  # small negative boost for aged tasks in safe regime

    # === UNCERTAINTY BOOST (only when slack > 0 and uncertainty high, to preempt risk) ===
    unc_boost = np.where((slack > 0.0) & (uncertainty > 0.15), 
                        np.clip(uncertainty * 0.25, 0.0, 0.12), 0.0)
    norm_unc = normalize_robust(unc_boost)
    unc_term = -0.07 * norm_unc

    # Combine with lexicographic weighting: urgency dominates, then SEER, then CPDD, etc.
    score = urgency_term + seer_term + cpdd_term + wait_term + unc_term

    # Final sanitization: ensure finite output, shape (N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
