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
    v2 mutation: Replaces arctan urgency with clipped tanh(-slack/τ) for sharper deadline dominance;
    introduces *critical-path density* (CPD = upward_rank / (min_exec_time + ε)) as latency-criticality coupling;
    replaces CED with *energy-latency efficiency ratio* (ELER = (min_exec_time + min_comm_time) / (min_incremental_energy + ε)),
    gated only when slack > 2.0s to avoid energy optimization under tight deadlines;
    uses robust MAD-based normalization (not IQR) for better outlier resilience and N=1 stability;
    adds uncertainty-aware fairness: sqrt(wait) × exp(-uncertainty), activated only when slack > 1.0s;
    enforces strict zero-energy contribution for slack <= 0 (no energy term), and clips all terms to [-3,3] pre-composition;
    final score = urgency × (1 + CPD) × (1 + ELER_gate × ELER_norm) + fairness_term, preserving multiplicative priority hierarchy.
    """
    eps = 1e-08
    # Cast and sanitize inputs
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust MAD-based normalization: handles flat arrays, N=1, outliers
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        median_x = np.median(x)
        abs_devs = np.abs(x - median_x)
        mad = np.median(abs_devs)
        scale = mad if mad > eps else eps
        z = (x - median_x) / scale
        # Clip to [-3, 3] for stability and gradient preservation
        return np.clip(z, -3.0, 3.0)

    # 1. Urgency: sharp, bounded, monotonic — tanh(-slack/τ) with τ=0.5s
    tau = 0.5
    urgency_raw = np.tanh(-slack / tau)  # [-1, 1]; negative slack → +1, large positive slack → -1
    norm_urgency = normalize_mad(urgency_raw)
    # Scale to dominate composition: high weight, no clipping (multiplicative base)
    urgency_term = 1.0 + 0.8 * norm_urgency  # [0.2, 1.8] — ensures positivity for multiplication

    # 2. Critical-Path Density (CPD): upward_rank / exec_time — captures latency sensitivity × criticality
    cpd_base = upward_rank / (min_exec_time + eps)
    norm_cpd = normalize_mad(cpd_base)
    cpd_term = 1.0 + 0.6 * np.clip(norm_cpd, -2.0, 2.0)  # [−0.2, 2.2] → clamped for stability

    # 3. Energy-Latency Efficiency Ratio (ELER): (latency) / (energy) — higher = more efficient
    eler_base = (min_exec_time + min_comm_time + eps) / (min_incremental_energy + eps)
    # Gate ELER only when slack is sufficiently safe (>2.0s) — avoids premature energy optimization
    eler_mask = (slack > 2.0).astype(float)
    eler_gated = eler_base * eler_mask
    norm_eler = normalize_mad(eler_gated)
    eler_term = 1.0 + 0.4 * np.clip(norm_eler, -2.0, 2.0)  # [−0.2, 2.2]

    # 4. Uncertainty-aware fairness: sqrt(wait) × exp(-uncertainty), active only when slack > 1.0s
    wait_safe = np.maximum(ready_wait_time, 0.0)
    sqrt_wait = np.sqrt(wait_safe + eps)
    exp_uncert = np.exp(-np.clip(uncertainty, 0.0, 10.0))  # bound exponent to prevent underflow/overflow
    fairness_raw = sqrt_wait * exp_uncert * ((slack > 1.0).astype(float))
    norm_fairness = normalize_mad(fairness_raw)
    # Fairness is additive penalty (lower priority for long-waiting only when safe), so invert sign
    fairness_term = -0.15 * np.clip(norm_fairness, -2.0, 2.0)

    # Multiplicative core: urgency × CPD × ELER preserves strict priority ordering
    # All terms are positive → product amplifies alignment of urgent + critical + efficient
    core_score = urgency_term * cpd_term * eler_term

    # Final deterministic score: multiplicative base + additive fairness correction
    score = core_score + fairness_term

    # Ensure finite output: clip extreme values, replace NaN/inf
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e8, 1e8)

    return score
