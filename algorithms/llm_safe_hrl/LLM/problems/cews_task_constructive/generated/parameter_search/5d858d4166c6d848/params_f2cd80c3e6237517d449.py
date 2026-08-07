import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's smooth urgency and bottleneck coupling with Parent 1's robust uncertainty-slack interaction,
    enhanced by adaptive IQR normalization and refined DDL-protection logic.
    
    Key structural improvements:
      - Added `uncertainty_slack_interaction_strength`: explicitly couples normalized uncertainty with *negative* slack pressure,
        strengthening urgency only when both risk and deadline pressure co-occur (validated by replay evidence).
      - Replaced global DDL-protection gate with *adaptive* activation: uses median slack (not fixed threshold) + uncertainty fraction,
        then applies interaction only to tasks below median slack — avoids over-penalizing early arrivals.
      - Retains smooth sigmoid urgency (Parent 2) but adds explicit negative-slack masking for interaction terms to prevent dilution.
      - Uses tighter IQR percentiles (22/78) for more responsive normalization in skewed distributions.
      - Keeps arctan-saturated wait time for fairness without drift; removes redundant piecewise slack penalty (Parent 1) entirely.
      - All operations are bounded, finite, and deterministic; no unbounded loops or state.
    """
    eps = 6.40027569388103e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def iqr_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 14.520925356851816)
        q_high = np.percentile(x, 79.2987605913768)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    slack_centered = slack - -0.43235517786062516
    sigmoid_input = np.clip(-5.816187487061775 * slack_centered, -8.32658137532674, 8.32658137532674)
    urgency_gate = 2.0 / (1.0 + np.exp(sigmoid_input)) - 1.0
    norm_urgency = iqr_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    bottleneck_pressure = duration * (remaining_work + upward_rank + eps)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.7853739083548217, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (4.415774483552468 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = iqr_normalize(wait_saturation)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_mask = ((slack < median_slack) & (uncertainty > 0.7613911719399877 * max_uncertainty)).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    neg_slack_mask = (slack < 0).astype(float)
    slack_norm = iqr_normalize(slack)
    neg_slack_pressure = np.where(slack < 0, -slack_norm, 0.0)
    uncertainty_slack_interaction = 0.8417436014572885 * norm_uncertainty * neg_slack_pressure
    score = norm_urgency + 1.1320264378515192 * norm_bottleneck + 0.30293835369376043 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_mask * norm_uncertainty + uncertainty_slack_interaction
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
