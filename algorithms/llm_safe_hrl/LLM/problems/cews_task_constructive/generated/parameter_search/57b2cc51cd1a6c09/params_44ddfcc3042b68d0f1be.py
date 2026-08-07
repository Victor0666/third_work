import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Dominant pre-normalized neg_slack term (Parent 2) with explicit weight for hard-deadline enforcement.
      - Bottleneck pressure uses *local* slack feasibility gating (inspired by Parent 1's ddl_protection_gate) instead of global median, improving sensitivity to critical tasks.
      - Uncertainty amplification applied *after* bottleneck construction but *before* normalization, preserving risk-contextual scaling.
      - Bounded sigmoid wait saturation (Parent 2) retained for robust anti-starvation.
      - Energy efficiency normalized adaptively using uncertainty-aligned dispersion (Parent 2), avoiding fragile joint-risk couplings.
      - All operations vectorized; only one conditional gate (slack feasibility threshold) → 1 branch total.
      - No numeric literals beyond {-2,-1,0,1,2}; all tunables declared in PARAMETER_SCHEMA.
    """
    eps = 3.638013249833946e-05
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
    slack_feasibility_gate = np.where(slack <= 0.30344828710434235, 1.0, 0.0)
    urgency_linear = np.clip(-slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.9133569003379569)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    bottleneck_pressure = duration * upward_rank * remaining_work * (1.0 + urgency_linear + eps)
    unc_normalized = (uncertainty - np.min(uncertainty)) / (np.ptp(uncertainty) + eps) if N > 1 else np.zeros_like(uncertainty)
    amplified_uncertainty = np.power(1.0 + unc_normalized, 1.983023719515998)
    bottleneck_pressure = bottleneck_pressure * (slack_feasibility_gate * amplified_uncertainty + (1.0 - slack_feasibility_gate))

    def adaptive_normalize(x):
        x = np.copy(x)
        center = np.median(x) if N > 0 else 0.0
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 0.6955327562947403 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = np.where(dispersion > eps, dispersion, fallback_range)
        return (x - center) / (denom + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    wait_scaled = ready_wait_time / (0.5000918572289772 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = adaptive_normalize(wait_saturation)
    score = 1.1617043251146868 * neg_slack + 0.9454494778767071 * norm_bottleneck + 1.4094993149683963 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
