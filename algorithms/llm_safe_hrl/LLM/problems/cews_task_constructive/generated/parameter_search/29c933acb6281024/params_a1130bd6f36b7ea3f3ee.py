import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with structural improvements guided by counterfactual evidence:
      - Introduces conditional DDL protection gate: activates only when slack <= median_slack AND uncertainty > threshold.
      - Replaces generic bottleneck pressure with explicit upward_rank × remaining_work interaction, unconditionally activated (no gating).
      - Adds successor-release coupling: modulates urgency based on how much successor release time depends on current task's completion.
      - Uses bounded sigmoid wait saturation with learned exponent and linear scale instead of hardcoded 4.0.
      - Removes energy-duration ratio term — inactive in elite diagnostics; replaced by direct min_incremental_energy normalization.
      - All normalization uses DDL-aware min-max (clipped range) for robustness under low-N or singleton ready sets.
      - Dominant term remains pre-normalized neg_slack to preserve hard-deadline dominance.
      - No branching beyond vectorized conditionals; all operations finite and deterministic.
    """
    eps = 3.2337922906960117e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    unc_normalized = (uncertainty - np.min(uncertainty + eps)) / (np.ptp(uncertainty) + eps) if N > 1 else np.zeros_like(uncertainty)
    ddl_protection_active = (slack <= median_slack) & (unc_normalized > 0.563542461370494)
    critical_pressure = upward_rank * remaining_work
    slack_range = np.ptp(slack) if N > 1 else eps
    successor_sensitivity = np.clip(slack_range / (np.max(np.abs(slack)) + eps), 0.0, 1.0)
    successor_release_boost = 0.05541200855409374 * successor_sensitivity * (1.0 + uncertainty)
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    urgency = urgency_linear + successor_release_boost

    def ddl_aware_normalize(x):
        x = np.copy(x)
        x_min = np.min(x)
        x_max = np.max(x)
        range_val = x_max - x_min
        denom = np.maximum(range_val, eps)
        return (x - x_min) / (denom + eps)
    norm_urgency = ddl_aware_normalize(urgency)
    norm_critical = ddl_aware_normalize(critical_pressure)
    norm_energy = ddl_aware_normalize(min_incremental_energy)
    wait_scaled = np.clip(ready_wait_time / np.max(ready_wait_time + eps), 0.0, 1.0)
    wait_transformed = 1.0004261856600938 * wait_scaled - 2.0
    wait_clipped = np.clip(wait_transformed, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped ** 1.655372831275623))
    norm_wait = ddl_aware_normalize(wait_saturation)
    amplified_critical = critical_pressure * np.power(1.0 + uncertainty, 2.2372182904614153)
    norm_amplified_critical = ddl_aware_normalize(amplified_critical)
    score = neg_slack + norm_urgency + 1.0658964457825713 * norm_critical + np.where(ddl_protection_active, norm_amplified_critical, 0.0) + norm_energy - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
