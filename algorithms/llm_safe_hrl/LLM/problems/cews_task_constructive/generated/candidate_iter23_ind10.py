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
    # Robust input sanitization: handle NaN, inf, and extreme values
    eps = 1e-08
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=eps, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=eps, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=eps, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    # Unified robust normalization: MAD-based with minmax fallback for N=1 and outliers
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        z = (x - median_x) / mad
        z_clipped = np.clip(z, -3.0, 3.0)  # Wider clipping than v1 for stability
        z_min, z_max = np.min(z_clipped), np.max(z_clipped)
        if z_max - z_min < eps:
            return np.zeros_like(z_clipped)
        return (z_clipped - z_min) / (z_max - z_min)  # [0,1] range avoids sign flip ambiguity

    # Term 1: Urgency — monotonic, differentiable slack penalty via tanh; high priority for negative slack
    # tanh(-slack/tau) → 1 when slack << 0, → 0 when slack >> 0; smooth transition at slack=0
    tau_urgency = 10.0
    urgency_raw = np.tanh(-slack / tau_urgency)
    norm_urgency = robust_normalize(urgency_raw)
    urgency_term = 1.0 - 0.9 * norm_urgency  # smaller score = higher priority → invert normalized urgency

    # Term 2: Critical-path density (CPD) — importance per unit latency; favors high-rank, low-latency tasks
    total_latency = min_exec_time + min_comm_time + eps
    cpd_base = upward_rank / total_latency
    norm_cpd = robust_normalize(cpd_base)
    cpd_term = 1.0 - 0.7 * norm_cpd  # prioritize high CPD (smaller score)

    # Term 3: Energy-aware critical path (CED) — energy efficiency weighted by deadline feasibility
    # Uses sigmoid gating σ(slack) to smoothly activate energy awareness only when slack >= 0
    ced_base = (upward_rank * remaining_work) / (min_incremental_energy + eps)
    ced_gate = 1.0 / (1.0 + np.exp(-slack / 5.0))  # smoother slope than v1 (tau=5 instead of implicit 1)
    ced_gated = ced_base * ced_gate
    norm_ced = robust_normalize(ced_gated)
    ced_term = 1.0 - 0.5 * norm_ced  # prioritize high CED (smaller score)

    # Term 4: Deadline violation penalty — additive, non-gated, scaled by uncertainty to penalize risky late tasks
    slack_violation = np.maximum(-slack, 0.0)
    penalty_base = slack_violation * (1.0 + np.clip(uncertainty, 0.0, 10.0))
    norm_penalty = robust_normalize(penalty_base)
    penalty_term = 0.8 * norm_penalty  # additive penalty → increases score (lowers priority)

    # Term 5: Fairness & aging — reward long-waiting tasks but attenuate by uncertainty to avoid risky assignments
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    exp_uncert = np.exp(-np.clip(uncertainty, 0.0, 5.0))  # softer attenuation than v1
    fairness_raw = sqrt_wait * exp_uncert
    norm_fairness = robust_normalize(fairness_raw)
    fairness_term = -0.2 * norm_fairness  # negative term lowers score for aged tasks

    # Combine terms multiplicatively for synergy (urgency × CPD × CED) + additively penalize violations & boost fairness
    core_score = urgency_term * cpd_term * ced_term
    score = core_score + penalty_term + fairness_term

    # Final safeguard: finite bounds and NaN/inf cleanup
    score = np.nan_to_num(score, nan=1e8, posinf=1e8, neginf=-1e8)
    score = np.clip(score, -1e7, 1e7)

    # Ensure shape (N,) explicitly
    return score.reshape(-1)
