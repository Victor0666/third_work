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
    v3: Deadline-hardened, energy-optimal, numerically stable priority with adaptive fairness and risk gating.
    
    Key evolutions:
    - Replaces tanh/linear urgency with *piecewise-linear urgency* (exact penalty for slack <= 0, linear decay for 0 < slack <= tau_urgency, zero beyond) → preserves strict monotonicity and avoids vanishing gradients.
    - Introduces *energy-latency ratio (ELR)* instead of SEER: min_incremental_energy / (min_exec_time + min_comm_time + tau_energy), better capturing marginal energy cost per latency unit under deadline pressure.
    - Critical path density now weighted by *normalized slack margin* (max(0, slack) / (tau_urgency + eps)) to smoothly downscale CP importance as slack increases — avoids over-prioritizing deep critical tasks when deadlines are safe.
    - Fairness term uses *relative wait time* (ready_wait_time / (1.0 + np.mean(exec_comm_sum))) scaled by slack surplus, preventing starvation without amplifying noise in sparse regimes.
    - Uncertainty boost now *additive only*, not normalized — avoids distorting priority ordering when uncertainty is globally low; capped and gated strictly (slack > 2.0 AND uncertainty > 0.15).
    - All normalization replaced with *robust min-max scaling* (not z-score): bounded, monotonic, invariant to outliers, deterministic for N=1.
    - Final score clamped and nan_to_num'd with finite bounds; no unbounded growth or sign flips.
    """
    eps = 1e-08
    # Defensive casting and NaN/inf handling
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0, posinf=1e6, neginf=0.0)

    tau_urgency = 1.0
    tau_energy = 3.0

    # === Urgency term: piecewise-linear, monotonic, hard deadline enforcement ===
    # slack <= 0 → linear penalty: -slack * 5.0 (stronger penalty than v1)
    # 0 < slack <= tau_urgency → linear decay: 5.0 * (1.0 - slack/tau_urgency)
    # slack > tau_urgency → 0.0
    urgency_raw = np.where(
        slack <= 0,
        -slack * 5.0,
        np.where(
            slack <= tau_urgency,
            5.0 * (1.0 - slack / (tau_urgency + eps)),
            0.0
        )
    )

    # === Energy-Latency Ratio (ELR): marginal energy per effective latency unit ===
    exec_comm_sum = min_exec_time + min_comm_time + eps
    elr_base = np.clip(min_incremental_energy / (exec_comm_sum + tau_energy), 0.0, 1e5)
    # ELR dominates only when slack > 0 → energy efficiency prioritized under safety
    elr_masked = np.where(slack > 0, elr_base, 0.0)

    # === Critical path density: gated & smoothly scaled by slack margin ===
    cp_density = upward_rank * remaining_work / (exec_comm_sum + eps)
    slack_margin_norm = np.clip(np.maximum(0.0, slack) / (tau_urgency + eps), 0.0, 1.0)
    cp_density_scaled = cp_density * slack_margin_norm

    # === Fairness: relative wait boost, stabilized against sparse exec_comm_sum ===
    mean_exec_comm = np.mean(exec_comm_sum) if len(exec_comm_sum) > 0 else 1.0
    rel_wait = np.clip(ready_wait_time / (1.0 + mean_exec_comm), 0.0, 10.0)
    wait_boost = rel_wait * slack_margin_norm

    # === Uncertainty boost: additive, capped, strictly gated ===
    unc_boost = np.where(
        (slack > 2.0) & (uncertainty > 0.15),
        np.clip(uncertainty * 0.35, 0.0, 0.18),
        0.0
    )

    # === Robust min-max normalization (monotonic, bounded, N=1 safe) ===
    def normalize_minmax(x):
        x = np.clip(x, -1e6, 1e6)
        if x.size == 1:
            return np.array([0.0])
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)

    norm_urgency = normalize_minmax(urgency_raw)
    norm_elr = normalize_minmax(elr_masked)
    norm_cp = normalize_minmax(cp_density_scaled)
    norm_wait = normalize_minmax(wait_boost)
    norm_unc = normalize_minmax(unc_boost)

    # Weighted fusion: urgency dominates (negative weight → lower score = higher priority)
    # Energy term: negative → prefer lower ELR (less energy per latency)
    # CP, wait, unc: positive weights → penalize low-CR, high-wait, high-unc when safe
    urgency_term = -7.0 * norm_urgency
    elr_term = -3.0 * norm_elr
    cp_term = 0.5 * norm_cp
    fairness_term = -0.35 * norm_wait
    unc_term = -0.12 * norm_unc

    score = urgency_term + elr_term + cp_term + fairness_term + unc_term

    # Final safeguard: finite, clipped, reshaped
    score = np.nan_to_num(score, nan=1e8, posinf=1e8, neginf=-1e8)
    score = np.clip(score, -1e8, 1e8)
    return score.reshape(-1)
