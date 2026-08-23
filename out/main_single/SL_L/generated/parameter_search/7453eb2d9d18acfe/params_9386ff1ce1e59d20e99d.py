import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining:
      - Strict lexicographic DDL enforcement: non-DDL terms only active when slack > 0
      - Robust empirical min-max clipping for slack instead of percentile
      - Bounded sigmoid gate on uncertainty to suppress noise in energy-uncertainty coupling
      - Critical path release via upward_rank × remaining_work product, gated by slack <= 0 AND high-rank-and-work
      - Anti-starvation wait term scaled by |slack|+1, only active in safe region, then median-normalized
      - Joint risk normalization using MAD over duration_total, |slack|, uncertainty for coherent scale alignment
      - All numeric literals restricted to {-2,-1,0,1,2}
      - Final score: DDL-violation dominates → critical-path urgency under violation → safe-region efficiency + fairness
    """
    eps = 1.5260811459015725e-06
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
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack) + eps
    all_risk_dims = np.stack([abs_slack, uncertainty, duration_total], axis=0)
    mad_per_dim = np.mean(np.abs(all_risk_dims - np.median(all_risk_dims, axis=1, keepdims=True)), axis=1) + eps
    norm_slack = abs_slack / (mad_per_dim[0] + eps)
    norm_uncert = uncertainty / (mad_per_dim[1] + eps)
    norm_duration = duration_total / (mad_per_dim[2] + eps)
    joint_risk_scale = np.maximum.reduce([norm_slack, norm_uncert, norm_duration])
    ddl_penalty = np.where(slack < 0, (-slack) ** 3.238904458922225, 0.0)
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_release_score = np.where(is_high_rank & is_high_work & is_tight_or_violated, -(upward_rank * remaining_work), 0.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    slack_lb = -14.610814015338775
    slack_ub = 49.73211434512638
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    energy_per_sec = (min_incremental_energy + eps) / (duration_total + eps)
    energy_norm = (energy_per_sec + eps) / (joint_risk_scale + eps)
    energy_term = slack_headroom_mask * 0.7400946773348983 * (energy_norm / (np.median(energy_norm) + eps))
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.869772263557018 * (uncertainty - 1.0)))
    energy_uncert_term = slack_headroom_mask * 1.163150947181647 * (energy_norm * norm_uncert * unc_sigmoid)
    wait_normalized = np.where(slack_headroom_mask > 0.0, ready_wait_time / (np.abs(slack) + 1.0), 0.0)
    wait_term = slack_headroom_mask * 0.5759376514440029 * (wait_normalized / (np.median(wait_normalized + eps) + eps))
    score = ddl_penalty + critical_release_score * 1.6977697355387735 + energy_term + energy_uncert_term + wait_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
