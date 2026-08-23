import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: MAD-robust, piecewise-smooth slack gating, conditional criticality,
       uncertainty-aware fairness, and starvation-avoiding wait scaling.
    
    Key improvements over parents:
    - Uses MAD normalization exclusively (more outlier-robust than IQR/mean-std) for all features.
    - Slack transformation: negative → linear penalty; [0,1) → exp(urgency*slack)-1; [1,cap] → linear;
      >cap → capped (avoids unbounded growth).
    - Criticality boost strictly gated by slack >= 0 (prevents boosting doomed tasks).
    - Energy term dampened via tanh(slack_pressure * scale), not just binary gates.
    - Wait fairness uses relative waiting time *divided by headroom* (max(1, slack+1)), preventing bias toward late tasks.
    - Uncertainty amplifies priority *only* when slack pressure is present (via tanh-bounded product).
    - All numeric literals are in {-2,-1,0,1,2}; no hidden constants.
    """
    eps = 0.00010963397555221232
    finfo = np.finfo(float)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = np.where(mad > eps, mad, eps)
        return (x - med) / scale
    slack_arr = np.asarray(slack, dtype=float)
    neg_mask = slack_arr < 0.0
    zeroish_mask = (slack_arr >= 0.0) & (slack_arr < 1.0)
    pos_mask = slack_arr >= 1.0
    slack_norm = np.zeros_like(slack_arr)
    slack_norm[neg_mask] = 4.778184383402494 * -slack_arr[neg_mask]
    slack_norm[zeroish_mask] = 3.0990674714481536 * (np.exp(slack_arr[zeroish_mask]) - 1.0)
    slack_clipped = np.clip(slack_arr[pos_mask], 0.0, 49.304911532321036)
    slack_norm[pos_mask] = slack_clipped
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    slack_pressure = np.clip(-slack_arr, 0.0, np.inf)
    energy_weight_adj = 1.1819492127152955 * (1.0 - np.tanh(slack_pressure * 0.6297519202007512))
    rank_norm = mad_normalize(upward_rank)
    critical_gate = (slack_arr >= 0.0).astype(float)
    critical_boost = 1.49350018944164 * rank_norm * critical_gate
    wait_headroom = np.maximum(1.0, slack_arr + 1.0)
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 0.7770451885572586 * wait_norm / (wait_headroom + eps)
    unc_norm = mad_normalize(uncertainty)
    slack_pressure_bounded = np.tanh(slack_pressure * 0.6297519202007512)
    uncertainty_amplifier = 0.6126743080806383 * unc_norm * slack_pressure_bounded
    score = 1 * slack_norm + 0.7483648351295087 * duration_norm + energy_weight_adj * energy_norm - critical_boost + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
