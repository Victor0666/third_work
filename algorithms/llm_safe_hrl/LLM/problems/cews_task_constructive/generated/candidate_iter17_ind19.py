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
    v2 synthesis: Combines Parent 2's robust rank-based normalization and adaptive sigmoid urgency
    with Parent 1's critical-path density ratio (CPDR) and violation-aware fairness gating.
    Key innovations:
    - Uses *rank-based percentile scaling* (Parent 2) for all terms to preserve ordinal stability.
    - Adaptive sigmoid urgency with steepness tuned to global slack pressure (Parent 2), but
      clamped to [-1,1] range for consistent dominance weight.
    - CPDR = upward_rank / (min_exec_time + min_comm_time + eps) as high-leverage bottleneck signal,
      normalized via rank scaling — replaces less stable density product.
    - SEER redefined as *soft-gated latency-to-energy ratio*: (exec+comm)/energy, multiplied by
      exp(-max(0,-slack)/tau_seer) to smoothly suppress energy optimization under violation.
    - Fairness term applied *only during slack violation* (slack < 0) as sqrt(wait)/max(1, -slack+eps),
      rank-normalized and bounded to prevent starvation without diluting urgency dominance.
    - Uncertainty penalizes communication time *only when slack > 0 and uncertainty is high*,
      avoiding over-penalization of critical late tasks.
    - Strict multiplicative hierarchy: urgency dominates → then CPDR boost → then SEER boost → then fairness/uncertainty additive corrections.
    - All operations protected against zeros, NaNs, infinities; deterministic and finite-valued.
    """
    eps = 1e-8
    # Cast inputs safely
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust rank-based normalization: maps x to [-1,1] preserving order
    def normalize_rank(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        sorted_x = np.sort(x)
        # Handle duplicates: use fractional rank
        ranks = np.searchsorted(sorted_x, x, side='left')
        # Normalize to [0,1], then map to [-1,1]
        ranks_norm = ranks / max(len(sorted_x) - 1, 1)
        return 2.0 * ranks_norm - 1.0

    # === URGENCY TERM (dominant, hard-DDL focused) ===
    # Adaptive tau: steeper when median slack is negative (tight deadlines)
    slack_median = np.median(slack)
    tau_urg = np.clip(1.0 + 0.5 * np.maximum(0.0, -slack_median), 0.3, 4.0)
    # Sigmoid urgency: near 1 when slack << 0, near 0 when slack >> 0 → invert for "smaller score = higher priority"
    urgency_raw = 1.0 / (1.0 + np.exp((slack + 1.0) / tau_urg))  # shift center for sharper transition
    norm_urgency = normalize_rank(urgency_raw)
    # Clamp to avoid extreme weights; scale so urgency_term ∈ [-5.0, 0.0] for strong dominance
    urgency_term = -5.0 * np.clip(norm_urgency, -1.0, 1.0)

    # === CRITICAL-PATH DENSITY RATIO (CPDR) ===
    # High upward_rank + low latency cost = high-leverage schedulable bottleneck
    latency_cost = min_exec_time + min_comm_time + eps
    cpdr_base = upward_rank / latency_cost
    norm_cpdr = normalize_rank(cpdr_base)
    cpdr_term = np.clip(norm_cpdr, -1.0, 1.0)
    # Multiplicative boost: enhances urgency term only where CPDR is high (positive contribution to priority score)
    cpdr_boost = 1.0 + 0.35 * np.maximum(cpdr_term, 0.0)

    # === SLACK-GATED ENERGY EFFICIENCY RATIO (SEER) ===
    # Prioritize energy efficiency only when headroom exists; smoothly fade under violation
    seer_base = latency_cost / (min_incremental_energy + eps)
    tau_seer = 1.5
    seer_gate = np.exp(-np.maximum(0.0, -slack) / tau_seer)  # decays as slack goes negative
    seer_gated = seer_base * seer_gate
    norm_seer = normalize_rank(seer_gated)
    seer_term = np.clip(norm_seer, -1.0, 1.0)
    seer_boost = 1.0 + 0.45 * np.maximum(seer_term, 0.0)

    # === FAIRNESS TERM (violation-only starvation prevention) ===
    # Only active when slack < 0: sqrt(wait) / max(1, -slack) → rewards waiting long when late
    fairness_raw = np.where(
        slack < 0,
        np.sqrt(np.maximum(ready_wait_time, 0.0) + eps) / np.maximum(-slack + eps, 1.0),
        0.0
    )
    norm_fairness = normalize_rank(fairness_raw)
    # Bounded positive correction (added to score), scaled to not override urgency
    fairness_term = 0.35 * np.clip(norm_fairness, 0.0, 1.0)

    # === UNCERTAINTY PENALTY (low-risk optimization only) ===
    # Only applied when slack > 0 AND uncertainty is in top 20% → avoids penalizing critical late tasks
    unc_thresh = np.percentile(uncertainty, 80) if uncertainty.size > 1 else np.max(uncertainty)
    unc_penalty_raw = np.where(
        (slack > 0.0) & (uncertainty > unc_thresh),
        min_comm_time * uncertainty * 0.15,
        0.0
    )
    norm_unc = normalize_rank(unc_penalty_raw)
    unc_term = 0.1 * np.clip(norm_unc, 0.0, 1.0)

    # === COMBINE WITH STRICT HIERARCHY ===
    # Multiplicative urgency dominance: urgency_term * cpdr_boost * seer_boost
    # Then additive fairness & uncertainty corrections (weaker influence)
    base_score = urgency_term * cpdr_boost * seer_boost
    score = base_score + fairness_term + unc_term

    # Final sanitization: ensure finite, deterministic, shape-(N,)
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    return score
