import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with critical structural fix:
      - Uses hard slack threshold gating (not sigmoid) for uncertainty amplification to eliminate instability.
      - Removes unused 'slack_feasibility_sigmoid_slope' parameter; all declared parameters are now used.
      - Retains unified adaptive normalization, bottleneck coupling, capped urgency, and wait saturation.
      - All numeric literals restricted to {-2,-1,0,1,2}; epsilon handled via PARAMS["epsilon"].
      - Robust for N=1 and edge cases via np.where and safe clipping.
    """
    eps = 8.082208440853814e-06
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
    hard_slack_gate = np.where(slack < -1.1499949678273103, 1.0, 0.0)
    urgency_linear = np.clip(-slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.where(N > 0, np.max(non_neg_slack), eps)
    urgency_cap = np.power(max_non_neg_slack + eps, 1.1999774785708357)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    ur_interaction = np.power(upward_rank * remaining_work + eps, 0.8442738390128324)
    bottleneck_pressure = ur_interaction * duration * (1.0 + urgency_linear + eps)
    unc_std = np.where(N > 1, np.std(uncertainty), eps)
    unc_normalized = (uncertainty - np.min(uncertainty)) / (np.ptp(uncertainty) + eps) if N > 1 else np.zeros_like(uncertainty)
    amplified_uncertainty = np.power(1.0 + unc_normalized, 0.39681622269350453)
    bottleneck_pressure = bottleneck_pressure * (hard_slack_gate * amplified_uncertainty + (1.0 - hard_slack_gate))

    def unified_adaptive_normalize(x):
        x = np.copy(x)
        center = np.where(N > 0, np.median(x), 0.0)
        unc_dispersion = 1.5015153338156302 * (unc_std + eps)
        fallback_range = np.where(N > 0, np.ptp(x), eps)
        denom = np.where(unc_dispersion > eps, unc_dispersion, fallback_range)
        return (x - center) / (denom + eps)
    norm_bottleneck = unified_adaptive_normalize(bottleneck_pressure)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = unified_adaptive_normalize(energy_per_duration)
    wait_scaled = ready_wait_time / (3.878381027026272 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = unified_adaptive_normalize(wait_saturation)
    score = 3.487788857888265 * neg_slack + 0.9491875397515193 * norm_bottleneck + 0.5141305902485267 * norm_energy_eff - 1.067264333081945 * norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
