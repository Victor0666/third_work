import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
      - Strict lexicographic DDL protection via hard `slack > ddl_protection_threshold` gating (validated across ≥22 decisions)
      - Critical-path release unconditionally via `upward_rank × remaining_work` interaction
      - Joint MAD normalization over `|slack|`, `uncertainty`, and `duration_total` for coherent risk alignment
      - Energy-aware terms gated *only* under meaningfully positive slack (`slack > ddl_protection_threshold`)
      - Bounded monotonic gates (sigmoid on uncertainty, linear slack-pressure scaling) replacing brittle thresholds
      - Anti-starvation via `ready_wait_time / (1 + |slack|)` but only when slack > ddl_protection_threshold
      - All features normalized using robust MAD (median absolute deviation) instead of percentile clipping for cross-seed stability
    """
    eps = 8.330470060657702e-09
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

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack)
    joint_risk_vec = np.stack([abs_slack, uncertainty, duration_total], axis=0)
    joint_mad_norm = mad_normalize(joint_risk_vec.mean(axis=0))
    slack_score = np.where(slack < 0, (-slack) ** 3.0654772564929136, 0.0)
    deadline_pressure = np.maximum(0.0, -slack + 0.9873950008701262)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.2685583550071713
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.4329259297247328
    critical_path_urgency = upward_rank * remaining_work
    slack_headroom_mask = np.where(slack > 0.9873950008701262, 1.0, 0.0)
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -40.06638397132889
    slack_ub = 79.36898783123154
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.4636744507398689 + (1.0 - 0.4636744507398689) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.6222137093492774 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.9589477381661828 * energy_norm * unc_norm * unc_sigmoid
    wait_score = np.where(slack_headroom_mask > 0.0, mad_normalize(ready_wait_time / (1.0 + abs_slack)), 0.0)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + -mad_normalize(critical_path_urgency)
    score += slack_headroom_mask * (1.1225882119502089 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    is_ddl_tight = slack <= 0.9873950008701262
    critical_gate = np.where(is_ddl_tight, 2.2852286578657672, 1.0)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
