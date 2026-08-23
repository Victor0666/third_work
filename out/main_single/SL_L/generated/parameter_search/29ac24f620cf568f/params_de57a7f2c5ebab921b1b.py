import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's stability & lexicographic gating with Parent 1's smooth slack-uncertainty coupling and starvation-aware waiting.
    Key novelties:
      - Starvation mitigation uses thresholded activation + relative urgency scaling by |slack| + eps
      - Energy terms decay exponentially with slack headroom using energy_decay_rate
      - Robust min-max normalization with explicit N=1 guard and tanh-based soft clamp for slack scaling
      - Piecewise rank weighting replaced by single linear interpolation (satisfies parameter count limit)
      - All non-DDL terms strictly gated off when slack <= 0; DDL-critical terms remain dominant.
    """
    eps = 5.436889556353205e-08
    mm_eps = 0.07809024946346385
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def minmax_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        x_min = np.min(x)
        x_max = np.max(x)
        denom = x_max - x_min + mm_eps
        normalized = (x - x_min) / denom
        return np.clip(normalized, 0.0, 1.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.3323879487909274, 0.0)
    slack_sign = np.tanh(slack / (np.abs(slack) + eps))
    unc_slack_coupling = np.where(slack < 0, uncertainty * -slack * (1.0 + 1.307505769116249 * np.abs(slack_sign)), 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    duration_risk = duration_total * uncertainty * 0.7950312424484467
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 4.007676764642018, 1.0)
    starvation_mask = np.where((ready_wait_time >= 2.0) & is_tight_or_violated, 1.0, 0.0)
    wait_urgency = np.where(is_tight_or_violated, ready_wait_time / (np.abs(slack) + eps), 0.0)
    wait_score = starvation_mask * minmax_normalize(wait_urgency)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    load_proxy = 1.0 + minmax_normalize(uncertainty)
    scaled_energy = min_incremental_energy * load_proxy
    energy_per_sec = scaled_energy / duration_total
    energy_eff_score = minmax_normalize(energy_per_sec)
    slack_centered = (slack - -5.808648334750231) / (12.823703158410499 - -5.808648334750231 + eps)
    slack_clamped = np.tanh(slack_centered * 2.0)
    slack_headroom_norm = np.clip((slack_clamped + 1.0) / 2.0, 0.0, 1.0)
    energy_decay = np.exp(-0.6751625880168924 * slack_headroom_norm)
    energy_final = slack_headroom_mask * energy_eff_score * energy_decay * 0.6751625880168924
    slack_lb = -5.808648334750231
    slack_ub = 12.823703158410499
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    rank_weight = 1.0 - slack_scaled * 0.8580373305887066
    rank_score = slack_headroom_mask * -minmax_normalize(upward_rank) * rank_weight
    unc_sigmoid = 1.0 / (1.0 + np.exp(-0.5003029492671925 * (uncertainty - 1.0)))
    energy_norm = minmax_normalize(scaled_energy)
    unc_norm = minmax_normalize(uncertainty)
    energy_uncertainty_score = slack_headroom_mask * 0.4686388716330151 * energy_norm * unc_norm * unc_sigmoid
    score = minmax_normalize(slack_score) + minmax_normalize(unc_slack_coupling) + minmax_normalize(duration_risk) + wait_score
    score += energy_final + rank_score + energy_uncertainty_score
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
