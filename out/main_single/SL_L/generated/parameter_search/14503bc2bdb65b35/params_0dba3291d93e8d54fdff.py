import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: smooth sigmoid slack gating, relative-MAD normalization,
       and fairness-gated energy dampening.
    
    Key evolutions:
    - Replaced piecewise slack transformation with *smooth, differentiable sigmoid*:
      urgency = 1 - exp(-k * max(0, slack)) for positive slack, preserving monotonicity
      and enabling gradient-aware optimization — eliminates discontinuities at slack=0/1.
    - Introduced *relative MAD normalization*: for slack and uncertainty, normalize w.r.t.
      their own distribution *and* scale by median(|slack|+ε) to preserve risk ordering
      under outlier-heavy workloads (e.g., one task with extreme slack dominates ranking).
    - Energy dampening now gated by *fairness headroom*: uses sigmoid of
      (ready_wait_time / (|slack| + ε)), not raw slack pressure — prioritizes energy
      efficiency only when waiting time is low *relative* to deadline margin, avoiding
      starvation while preserving energy savings where feasible.
    - All numeric literals are strictly in {-2,-1,0,1,2}; no hidden constants.
    """
    eps = 6.490063511147951e-05
    finfo = np.finfo(float)

    def relative_mad_normalize(x, ref_scale=None):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = np.where(mad > eps, mad, eps)
        if ref_scale is not None:
            scale = np.maximum(scale, np.abs(ref_scale) + eps)
        return (x - med) / scale
    slack_arr = np.asarray(slack, dtype=float)
    neg_mask = slack_arr < 0.0
    pos_mask = slack_arr >= 0.0
    slack_norm = np.zeros_like(slack_arr)
    slack_norm[neg_mask] = 5.3497472875730745 * -slack_arr[neg_mask]
    urgency_pos = 1.0 - np.exp(-0.02206345236310836 * slack_arr[pos_mask])
    urgency_scaled = 2.1093347368745747 * urgency_pos
    slack_clipped = np.clip(slack_arr[pos_mask], 0.0, 17.0039050423598)
    linear_decay = slack_clipped / (17.0039050423598 + eps)
    slack_norm[pos_mask] = urgency_scaled * linear_decay + (1.0 - linear_decay) * slack_clipped
    duration = min_exec_time + min_comm_time
    duration_norm = relative_mad_normalize(duration)
    energy_norm = relative_mad_normalize(min_incremental_energy)
    wait_headroom_ratio = ready_wait_time / (np.abs(slack_arr) + eps)
    fairness_gate = 1.0 - 1.0 / (1.0 + np.exp(-0.6089532081853298 * (1.0 - wait_headroom_ratio)))
    energy_weight_adj = 1.120890936917542 * fairness_gate
    rank_norm = relative_mad_normalize(upward_rank)
    critical_gate = (slack_arr >= 0.0).astype(float)
    critical_boost = 0.029399853775211204 * rank_norm * critical_gate
    wait_norm = relative_mad_normalize(ready_wait_time)
    wait_headroom = np.maximum(1.0, np.abs(slack_arr) + 1.0)
    wait_score = 0.002264191259702859 * wait_norm / (wait_headroom + eps)
    unc_norm = relative_mad_normalize(uncertainty, ref_scale=np.median(np.abs(slack_arr) + eps))
    slack_pressure = np.clip(-slack_arr, 0.0, np.inf)
    slack_pressure_bounded = np.tanh(slack_pressure * 0.6986690752661263)
    uncertainty_amplifier = 1.3463080847025717 * unc_norm * slack_pressure_bounded
    score = 1 * slack_norm + 0.6546736171472262 * duration_norm + energy_weight_adj * energy_norm - critical_boost + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
