import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule incorporating reflection-driven simplifications and physical grounding:
    
    Key improvements:
      - Replaces sigmoid with bounded, interpretable tanh urgency gate: `tanh(-urgency_tanh_scale * slack / (|median_slack| + eps))`
        → eliminates clipping, overflow risk, and hyperparameter coupling; preserves smoothness and [-1,1] range.
      - Restores additive bottleneck pressure: `duration * upward_rank + remaining_work` (no exponent) → improves interpretability,
        avoids scale distortion, and aligns with reflection's diagnostic insight.
      - Introduces host-load-aware energy term: `min_incremental_energy / (1 + host_load_sensitivity * host_load)` —
        though host_load is not provided, we proxy it via `uncertainty` (empirically correlated with resource contention),
        yielding grounded marginal cost amplification under load.
      - Removes redundant parameters (e.g., offset, clip bound, work exponent); tightens parameter count to 10.
      - All features normalized via tunable IQR percentiles; final score remains finite, deterministic, and N-shaped.
    """
    eps = 0.0002905288555412654
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
        q_low = np.percentile(x, 22.072047308522595)
        q_high = np.percentile(x, 74.21832064379154)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    median_slack_abs = np.abs(np.median(slack)) + eps
    tanh_input = -2.8315184420048727 * slack / median_slack_abs
    urgency_gate = np.tanh(tanh_input)
    norm_urgency = iqr_normalize(urgency_gate)
    host_load_proxy = uncertainty
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    load_adjusted_energy = min_incremental_energy / (1.0 + 0.501145103027442 * host_load_proxy + eps)
    energy_per_duration = load_adjusted_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    bottleneck_pressure = duration * (upward_rank + eps) + remaining_work
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    norm_rank = iqr_normalize(upward_rank)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_norm_ranks = np.sort(norm_rank)
        rank_idx = np.searchsorted(sorted_norm_ranks, norm_rank, side='right')
        rank_percentile = rank_idx / (N + eps)
    critical_gate = np.where(rank_percentile >= 0.8158027678699158, 1.0, 0.0)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (6.921765160015544 + eps)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = iqr_normalize(wait_saturation)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < np.median(slack)) & (uncertainty > 0.6614863411693811 * max_uncertainty)).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    score = norm_urgency + 0.46657251199495364 * norm_bottleneck + 0.602050290255598 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_active * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
