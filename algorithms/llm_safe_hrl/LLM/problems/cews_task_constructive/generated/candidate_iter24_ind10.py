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
    eps = 1e-08
    # Robust sanitization: ensure finite, non-zero denominators and meaningful bounds
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=eps, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=eps, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=eps, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        median_x = np.median(x)
        mad = np.median(np.abs(x - median_x)) + eps
        z = (x - median_x) / mad
        z_clipped = np.clip(z, -3.0, 3.0)
        z_min, z_max = np.min(z_clipped), np.max(z_clipped)
        if z_max - z_min < eps:
            return np.zeros_like(z_clipped)
        return (z_clipped - z_min) / (z_max - z_min)

    # Sharp deadline urgency: use linear penalty for slack < 0, sigmoid ramp for slack >= 0
    # Avoid tanh flattening — preserve monotonicity and sensitivity near DDL boundary
    urgency_raw = np.where(slack < 0, -slack * 2.0, 1.0 - 1.0 / (1.0 + np.exp(-slack / 2.0)))
    norm_urgency = robust_normalize(urgency_raw)
    urgency_term = 1.0 + 0.95 * norm_urgency  # Higher urgency → lower score (priority ↑)

    # Critical-path delay sensitivity: CPD = upward_rank / latency; higher CPD → higher priority
    total_latency = min_exec_time + min_comm_time + eps
    cpd_base = upward_rank / total_latency
    norm_cpd = robust_normalize(cpd_base)
    cpd_term = 1.0 + 0.75 * norm_cpd  # Preserve original monotonic direction: high CPD → high priority

    # Energy-awareness only under feasibility: CED = (upward_rank * remaining_work) / energy
    # Scale gating by normalized criticality to adapt sensitivity to workflow structure
    criticality_scale = np.clip(upward_rank / (np.median(upward_rank) + eps), 0.1, 10.0)
    ced_base = upward_rank * remaining_work / (min_incremental_energy + eps)
    # Adaptive gating: stronger energy emphasis when slack is positive *and* criticality is high
    ced_gate = 1.0 / (1.0 + np.exp(-(slack + 2.0 * np.log(criticality_scale + eps)) / 3.0))
    ced_gated = ced_base * ced_gate
    norm_ced = robust_normalize(ced_gated)
    ced_term = 1.0 + 0.6 * norm_ced  # High CED → high priority (energy-efficient path favored when feasible)

    # Hard deadline violation penalty: linear in lateness, amplified by uncertainty
    slack_violation = np.maximum(-slack, 0.0)
    penalty_base = slack_violation * (1.0 + np.clip(uncertainty, 0.0, 10.0))
    norm_penalty = robust_normalize(penalty_base)
    penalty_term = 1.2 * norm_penalty  # Stronger penalty for violations than prior versions

    # Wait-based fairness: prioritize long-waiting tasks, but attenuate only by low uncertainty
    # Ensures fairness without undermining urgency or CPD — avoids non-monotonic inversion
    wait_safe = np.maximum(ready_wait_time, 0.0)
    wait_term = np.sqrt(wait_safe + eps)
    # Only reduce wait priority if uncertainty is *high* (trust deficit), not low
    uncert_attenuation = np.clip(1.0 - uncertainty / (np.max(uncertainty + eps, initial=1.0)), 0.0, 1.0)
    fairness_raw = wait_term * uncert_attenuation
    norm_fairness = robust_normalize(fairness_raw)
    fairness_term = -0.25 * norm_fairness  # Slightly stronger fairness pull than v1

    # Core composition: multiplicative to enforce joint satisfaction of urgency, CPD, and CED
    core_score = urgency_term * cpd_term * ced_term
    # Additive penalties preserve dominance of hard constraints (penalty_term > fairness_term)
    score = core_score + penalty_term + fairness_term

    # Final safeguard: ensure finite, bounded output with deterministic shape
    score = np.nan_to_num(score, nan=1e8, posinf=1e8, neginf=-1e8)
    score = np.clip(score, -1e7, 1e7)
    return score.reshape(-1)
