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
    Hybrid priority rule v2: merges Parent 2's clipped z-score stability and relative-slack gating
    with Parent 1's deadline-aware energy fairness and criticality-weighted wait boosting.
    Key innovations:
      - Uses robust clipped z-score (±3σ) for all terms to ensure numerical stability
      - Introduces *slack-gated energy fairness*: energy penalty only activates when slack > 0,
        scaled by sigmoid(slack / duration) to smoothly engage energy minimization in feasible regime
      - Combines Parent 2's hard-thresholded urgency gating (30th-percentile relative slack)
        with Parent 1's explicit lateness penalty (neg_slack × risk-amplified weight)
      - Replaces raw wait_boost with *criticality-normalized starvation boost*:
        only activates when slack > slack_30pct, scaled by upward_rank and wait time
      - Adds uncertainty-weighted duration penalty *only for non-urgent tasks* to avoid over-penalizing
        critical path nodes under tight deadlines
      - All divisions epsilon-protected; nan/inf cleaned; shape enforced.
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
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    # Precompute stable base quantities
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = slack / task_duration
    slack_30pct = np.quantile(rel_slack, 0.3) if N > 1 else np.min(rel_slack)
    is_urgent = (rel_slack <= slack_30pct).astype(float)
    neg_slack = np.maximum(-slack, 0.0)

    # --- Urgency signal: hard-thresholded + explicit lateness penalty ---
    # Base urgency: exponential decay for non-urgent, unit baseline for urgent
    urgency_base = np.where(
        is_urgent,
        1.0 + np.maximum(0.0, -rel_slack) * 0.8,
        1.0 / (1.0 + np.maximum(0.0, rel_slack - slack_30pct) * 0.5 + eps)
    )
    # Lateness penalty: only active when slack < 0, amplified by uncertainty
    lateness_penalty = neg_slack * (6.0 + 2.5 * np.clip(uncertainty, 0.0, 2.0))
    urgency_score = urgency_base + lateness_penalty

    # --- Energy fairness: activated only when slack > 0, smooth sigmoid gate ---
    energy_activation = 1.0 / (1.0 + np.exp(-np.clip(slack / (task_duration + eps), -5.0, 5.0)))
    risk_adjusted_energy = min_incremental_energy * (1.0 + 0.4 * np.clip(uncertainty, 0.0, 1.0))
    energy_score = risk_adjusted_energy * energy_activation

    # --- Criticality-duration synergy: (upward_rank * duration) / energy, robustly bounded ---
    synergy_num = upward_rank * task_duration * (1.0 + uncertainty)  # risk-weighted
    synergy_denom = np.maximum(risk_adjusted_energy, eps)
    latency_criticality_synergy = np.divide(synergy_num, synergy_denom, out=np.full(N, 1e-6), where=synergy_denom != 0)
    latency_criticality_synergy = np.clip(latency_criticality_synergy, 1e-6, 1e6)

    # --- Starvation control: criticality-normalized wait boost, only when slack permits ---
    wait_boost = np.where(
        slack > slack_30pct * task_duration,  # slack > relative threshold in absolute time
        np.clip(
            (ready_wait_time / (np.maximum(np.median(ready_wait_time), eps) + eps)) *
            (upward_rank / (np.maximum(np.median(upward_rank), eps) + eps)),
            0.0, 0.5
        ),
        0.0
    )

    # --- Uncertainty-duration penalty: only for non-urgent tasks to avoid penalizing critical path ---
    time_uncertainty_penalty = np.where(
        is_urgent == 0,
        uncertainty * task_duration,
        0.0
    )

    # --- Robust z-score normalization (clipped ±3σ) for all terms ---
    def robust_zscore(x):
        mu = np.mean(x)
        std = np.std(x, ddof=0) + eps
        z = (x - mu) / std
        return np.clip(z, -3.0, 3.0)

    urgency_norm = robust_zscore(urgency_score)
    energy_norm = robust_zscore(energy_score)
    synergy_norm = robust_zscore(latency_criticality_synergy)
    wait_norm = robust_zscore(wait_boost)
    work_norm = robust_zscore(remaining_work)
    unc_norm = robust_zscore(uncertainty)
    time_uncertainty_norm = robust_zscore(time_uncertainty_penalty)

    # --- Final weighted score: smaller = higher priority ---
    # Weights emphasize urgency (-), synergy (-), and fairness (+ energy only when safe)
    score = (
        -2.7 * urgency_norm
        - 2.0 * synergy_norm
        + 0.35 * energy_norm
        + 0.25 * unc_norm
        + 0.2 * time_uncertainty_norm
        - 0.3 * wait_norm  # negative because wait_boost increases priority
        + 0.12 * work_norm
    )

    # Final cleanup: handle nan/inf, clamp extremes, enforce shape
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
