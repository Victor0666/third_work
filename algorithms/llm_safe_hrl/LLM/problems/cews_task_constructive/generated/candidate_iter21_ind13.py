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
    v2 self-evolution: Fixes over-aggressive gating and multiplicative noise;
    replaces non-monotonic pressure with additive, monotonic slack penalty;
    restores energy-awareness under *feasible* deadlines (slack > 0.0) using CED;
    uses robust MAD+minmax normalization for stability at N=1 and outlier resilience;
    introduces deadline-feasibility-aware energy efficiency: only activates CED when slack >= 0,
    and scales it smoothly via sigmoid(slack) to avoid step-function artifacts;
    removes fairness slack decay (causes non-monotonicity) and replaces with wait-based urgency
    attenuated by uncertainty *only*, preserving monotonicity in slack;
    all terms bounded pre-composition and final score clipped to finite range.
    """
    eps = 1e-08
    # Safe casting and nan/inf handling
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def normalize_mad_minmax(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        median_x = np.median(x)
        abs_devs = np.abs(x - median_x)
        mad = np.median(abs_devs)
        scale = mad if mad > eps else eps
        z = (x - median_x) / scale
        # Further stabilize with minmax clipping to prevent outlier skew
        z_clipped = np.clip(z, -2.0, 2.0)
        # Normalize to [-1, 1] range for balanced contribution
        z_min, z_max = np.min(z_clipped), np.max(z_clipped)
        if z_max - z_min < eps:
            return np.zeros_like(z_clipped)
        return 2.0 * (z_clipped - z_min) / (z_max - z_min) - 1.0

    # Urgency: tanh(-slack/τ) with τ=0.5 → sharper than linear, smoother than τ=0.3
    tau_urgency = 0.5
    urgency_raw = np.tanh(-slack / tau_urgency)
    norm_urgency = normalize_mad_minmax(urgency_raw)
    urgency_term = 1.0 + 0.85 * norm_urgency  # dominant base priority

    # Critical-path density: communication-aware, robust denominator
    cpd_base = upward_rank / (min_exec_time + min_comm_time + eps)
    norm_cpd = normalize_mad_minmax(cpd_base)
    cpd_term = 1.0 + 0.65 * norm_cpd  # reinforces latency-critical tasks

    # Energy efficiency: CED = (upward_rank * remaining_work) / energy, activated smoothly
    # Only meaningful when slack >= 0; smoothly gated by sigmoid(slack) ∈ [0.5, 1.0] for slack ∈ [0, ∞)
    ced_base = upward_rank * remaining_work / (min_incremental_energy + eps)
    ced_gate = 1.0 / (1.0 + np.exp(-slack))  # sigmoid(slack): 0.5 at slack=0, →1 as slack→∞
    ced_gated = ced_base * ced_gate
    norm_ced = normalize_mad_minmax(ced_gated)
    ced_term = 1.0 + 0.45 * norm_ced  # promotes energy-efficient scheduling *only* when feasible

    # Monotonic slack penalty: additive, linear-in-violation, zero when slack ≥ 0
    # Avoids non-monotonicity: penalty = max(0, -slack) * (1 + uncertainty)
    slack_violation = np.maximum(-slack, 0.0)
    penalty_base = slack_violation * (1.0 + np.clip(uncertainty, 0.0, 5.0))
    norm_penalty = normalize_mad_minmax(penalty_base)
    penalty_term = 0.9 * norm_penalty  # strong corrective term for violations

    # Fairness: wait-based urgency, attenuated *only* by uncertainty (no slack decay → monotonic)
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    exp_uncert = np.exp(-np.clip(uncertainty, 0.0, 10.0))
    fairness_raw = sqrt_wait * exp_uncert
    norm_fairness = normalize_mad_minmax(fairness_raw)
    fairness_term = -0.18 * norm_fairness  # slight bias toward long-waiting tasks

    # Core multiplicative priority (deadline-driven) + additive corrections
    core_score = urgency_term * cpd_term * ced_term
    score = core_score + penalty_term + fairness_term

    # Final sanitization: ensure finite, bounded, deterministic output
    score = np.nan_to_num(score, nan=1e8, posinf=1e8, neginf=-1e8)
    score = np.clip(score, -1e7, 1e7)
    return score
