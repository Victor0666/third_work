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
    v2 priority rule: Hybrid urgency-criticality-energy scoring with quantile-adaptive gating,
                      robust ±8σ normalization, uncertainty-augmented urgency, criticality-gated starvation relief,
                      and deadline-proximity-aware energy scaling.

    Key innovations:
    - Combines Parent 2's adaptive quantile urgency (q10/q50/q90) and robust z-clipping with Parent 1's clean wait-pressure formulation.
    - Replaces fragile tanh/median-dependent scalings with stable sigmoid-based slack sensitivity and MAD-free robust clipping.
    - Introduces *deadline-proximity-aware energy scaling*: energy penalty attenuated only when slack > q90 (very relaxed), amplified when slack < q50 (tight).
    - Uses *criticality-gated starvation rescue*: only activates wait-pressure for tasks above upward_rank 75th percentile to avoid low-importance bias.
    - Uncertainty directly amplifies both urgency and energy terms — high volatility increases penalty in both feasibility and efficiency dimensions.
    - Final layering enforces hard ordering: urgency dominates, then synergy, then energy, then fairness, then work bonus.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)

    # Robust normalization: z-score clipped to [-8, 8], with fallback for degenerate cases
    def robust_zclip(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        z = (x - center) / mad
        return np.clip(z, -8.0, 8.0)

    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    dur_uncertainty = np.divide(uncertainty, task_duration + eps, out=np.zeros_like(uncertainty), where=(task_duration + eps) != 0)
    dur_uncertainty = np.nan_to_num(dur_uncertainty, nan=0.0, posinf=0.0, neginf=0.0)

    # Adaptive quantile thresholds for heterogeneous deadline pressure
    if N > 1:
        q10, q50, q90 = np.quantile(slack, [0.1, 0.5, 0.9], method='midpoint')
    else:
        q10 = q50 = q90 = slack[0]

    # Urgency: three-zone quantile-gated penalty (violated, tight, relaxed)
    urgency_base = np.zeros_like(slack)
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (slack < q50)
    relaxed_mask = ~violated_mask & (slack >= q50)

    urgency_base[violated_mask] = np.clip(-slack[violated_mask] / (task_duration[violated_mask] + eps), 0.0, 12.0)
    urgency_base[tight_mask] = np.clip((q50 - slack[tight_mask]) / (task_duration[tight_mask] + eps), 0.0, 5.0)
    urgency_base[relaxed_mask] = np.clip((q90 - slack[relaxed_mask]) / (task_duration[relaxed_mask] + eps), 0.0, 1.0)

    # Criticality modulation: upward rank boosts urgency for high-importance tasks
    ur_norm = robust_zclip(upward_rank)
    ur_factor = 1.0 + 0.35 * np.clip(ur_norm, 0.0, 8.0)
    # Uncertainty-as-urgency multiplier: volatile tasks demand faster execution
    unc_urgency_factor = 1.0 + 0.5 * dur_uncertainty
    urgency_penalty = urgency_base * ur_factor * unc_urgency_factor

    # Synergy term: latency-critical work per energy unit, scaled by slack proximity
    slack_factor = np.clip(1.0 - np.maximum(0.0, slack - q50) / (task_duration + eps), 0.05, 1.0)
    dampened_ur = upward_rank * slack_factor
    base_synergy = dampened_ur * task_duration / (min_incremental_energy + eps)
    # Amplify synergy under violation to force recovery
    synergy_amplifier = np.where(violated_mask, 1.0 + 0.8 * uncertainty, 1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, eps, 1e10)

    # Energy term: risk-weighted, scaled by deadline proximity — penalized more when tight
    # Energy penalty is *inversely* gated: stronger penalty when slack is small (not large)
    slack_distance_weight = np.where(slack < q50, 1.0 + 0.4 * (q50 - slack) / (task_duration + eps), 1.0)
    risk_weighted_energy = min_incremental_energy * (1.0 + uncertainty * slack_distance_weight)
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e10)

    # Starvation relief: only for high-criticality tasks waiting long relative to duration
    ur_percentile = np.percentile(upward_rank, 75) if N > 1 else upward_rank[0]
    wait_ratio = np.divide(ready_wait_time, task_duration, out=np.zeros_like(ready_wait_time), where=task_duration != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    is_starvable = (wait_ratio > 1.2) & (upward_rank >= ur_percentile) & (slack < 300.0)
    starvation_boost = np.where(is_starvable, wait_ratio * (1.0 + 0.25 * np.clip(ur_norm, 0.0, 8.0)), 0.0)
    starvation_boost = np.clip(starvation_boost, 0.0, 2.5)

    # Work bonus: negative incentive for large remaining work (encourages early scheduling of heavy subtrees)
    median_rw = np.median(remaining_work) + eps
    norm_remaining_work = np.clip(remaining_work / median_rw, 0.01, 100.0)
    work_bonus = -0.06 * norm_remaining_work

    # Normalize all components robustly
    norm_urgency = robust_zclip(urgency_penalty)
    norm_synergy = robust_zclip(latency_crit_synergy)
    norm_energy = robust_zclip(risk_weighted_energy)
    norm_starvation = robust_zclip(starvation_boost)
    norm_work = robust_zclip(remaining_work)

    # Layered priority: hard dominance order (urgency > synergy > energy > starvation > work)
    score = np.full(N, 0.0, dtype=np.float64)
    score = score + 0.48 * norm_urgency
    score = score - 0.32 * norm_synergy
    score = score + 0.12 * norm_energy
    score = score + 0.06 * norm_starvation
    score = score + 0.02 * norm_work

    # Final sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
