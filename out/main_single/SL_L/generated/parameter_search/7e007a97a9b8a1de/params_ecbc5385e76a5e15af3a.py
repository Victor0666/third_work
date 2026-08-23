import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Novel priority rule: deadline-risk-gated criticality + uncertainty-aware fairness.
    
    Key innovations:
    - Slack is transformed via piecewise smooth function: strong penalty for negative slack,
      exponential urgency for small positive slack (< 1s), linear decay beyond (capped).
    - Critical path leverage is *conditional*: only applied when slack >= 0, avoiding over-prioritizing 
      doomed tasks.
    - Energy term is dampened under high deadline pressure (via slack-based gate).
    - Wait fairness uses relative waiting time scaled by remaining slack headroom to avoid biasing late tasks.
    - Uncertainty interacts multiplicatively with slack pressure (not raw slack) for bounded risk amplification.
    - All features use MAD-normalization (more robust than mean/std) with epsilon safeguards.
    """
    eps = 0.00020144900338927852
    finfo = np.finfo(float)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = mad if mad > eps else eps
        return (x - med) / scale
    slack_arr = np.asarray(slack, dtype=float)
    neg_mask = slack_arr < 0.0
    zeroish_mask = (slack_arr >= 0.0) & (slack_arr < 1.0)
    pos_mask = slack_arr >= 1.0
    slack_norm = np.zeros_like(slack_arr)
    slack_norm[neg_mask] = 4.709498113050043 * -slack_arr[neg_mask]
    slack_norm[zeroish_mask] = 1.8309655398048519 * (np.exp(slack_arr[zeroish_mask]) - 1.0)
    slack_norm[pos_mask] = np.clip(slack_arr[pos_mask], 0.0, 16.836108690143543)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    slack_pressure = np.clip(-slack_arr, 0.0, np.inf)
    energy_weight_adj = 0.2665051285254927 * (1.0 - np.tanh(slack_pressure * 0.18727626814870046))
    rank_norm = mad_normalize(upward_rank)
    critical_gate = (slack_arr >= 0.0).astype(float)
    critical_boost = 0.7208726488346975 * rank_norm * critical_gate
    wait_headroom = np.maximum(1.0, slack_arr + 1.0)
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 0.6979937455056485 * wait_norm / (wait_headroom + eps)
    unc_norm = mad_normalize(uncertainty)
    slack_pressure_bounded = np.tanh(slack_pressure * 0.18727626814870046)
    uncertainty_amplifier = 0.13533571568123506 * unc_norm * slack_pressure_bounded
    score = slack_norm + 1.0631857417102621 * duration_norm + energy_weight_adj * energy_norm - critical_boost + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
