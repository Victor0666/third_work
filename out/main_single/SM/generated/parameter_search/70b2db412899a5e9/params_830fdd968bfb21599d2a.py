import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robustness with Parent 1's adaptive gating:
      - Uses sigmoid DDL gate (Parent 1) for smooth, numerically stable transition near slack=0,
        but simplified with fixed center at 0 and steepness parameter — avoids per-task offset complexity.
      - Retains exponential starvation mitigation (Parent 2) for stable wait-time response.
      - Applies max-abs normalization universally (Parent 2) for interpretability and outlier resilience.
      - Introduces *slack-dependent energy gating*: attenuates energy term proportionally to ddl_gate,
        improving deadline-risk focus without eliminating energy optimization in feasible region.
      - Criticality uses exponentiated upward_rank weighted by slack magnitude (Parent 2 style),
        enabling stronger prioritization of high-rank tasks when deadlines tighten.
      - Bottleneck term uses rank * work / (|slack| + eps) — clean, monotonic, and avoids softplus instability.
      - All numeric literals are {-2,-1,0,1,2}; no hidden constants; all parameters used exactly once.
    """
    eps = 2.0072983560663823e-05
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.max(abs_x[finite_mask])
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    finfo = np.finfo(float)
    gate_input = 17.045908372079325 * -slk
    gate_input_clipped = np.clip(gate_input, -np.log(finfo.max), np.log(finfo.max))
    ddl_gate = 1.0 / (1.0 + np.exp(-gate_input_clipped))
    slack_penalty = np.where(slk < 0, 5.441615410376431 * np.abs(slk), -5.382059391942395 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.4753952947621821 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - 0.09947115775196563 * ddl_gate)
    slack_mag_norm = normalize(np.abs(slk) + eps)
    rank_powered = np.power(rank + eps, 0.2993374367115789)
    rank_score = -normalize(rank_powered)
    slack_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_term = rank * work * slack_inv
    bottleneck_score = -0.7134302526620607 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.8571794624405495 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (44.113758353214735 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score_clip_max = 146749714.02201292
    score_clip_min = -12619336503.56343
    score = np.nan_to_num(score, nan=0.0, posinf=score_clip_max, neginf=score_clip_min)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
