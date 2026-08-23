import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's stability with novel starvation modeling:
      - Decoupled per-feature MAD normalization (robust, no cross-contamination)
      - Hard DDL-protection gate + smooth tanh coupling (strict feasibility-first)
      - Novel starvation urgency exponentiation: wait_ratio^p strengthens priority for extreme waiting cases
      - Critical path release decoupled from slack normalization to avoid dilution under deadline pressure
      - All gates and interactions bounded, differentiable, and numerically safeguarded
    """
    eps = 1.814293634497523e-06
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
        x_med = np.median(x)
        x_mad = np.mean(np.abs(x - x_med)) + eps
        return (x - x_med) / (x_mad * 1.3352947521649297 + eps)
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack) + eps
    duration_norm = mad_normalize(duration_total)
    slack_norm = mad_normalize(abs_slack)
    unc_norm = mad_normalize(uncertainty)
    energy_norm = mad_normalize(min_incremental_energy)
    rank_norm = mad_normalize(upward_rank)
    work_norm = mad_normalize(remaining_work)
    wait_norm = mad_normalize(ready_wait_time)
    slack_sign = np.tanh(slack / (np.abs(slack) + eps))
    slack_penalty = np.where(slack < 0, -slack * (1.0 + 1.273890125204824 * unc_norm * np.abs(slack_sign) ** 0.7947721803602533), 0.0)
    critical_release_raw = upward_rank * remaining_work
    critical_release_score = -critical_release_raw * 0.5736583936273696
    ddl_safe_mask = np.where(slack >= 0.0, 1.0, 0.0)
    ramp_half = 2.5352009532851776 / 2.0
    gate_center = 0.6339409420032732
    gate_low = gate_center - ramp_half
    gate_high = gate_center + ramp_half
    slack_gate = np.clip((slack - gate_low) / (ramp_half * 2.0 + eps), 0.0, 1.0)
    safe_gate = ddl_safe_mask * slack_gate
    energy_score = safe_gate * energy_norm * 0.147464817513223
    energy_uncertainty_score = safe_gate * energy_norm * unc_norm * 1.298739872032959
    median_wait = np.median(ready_wait_time) + eps
    wait_ratio = np.clip(ready_wait_time / median_wait, 0.0, 2.0)
    wait_urgency = wait_ratio ** 0.963791035501274
    wait_score = ready_wait_time * (1.0 + np.clip(-slack / (0.6339409420032732 + eps), 0.0, 2.0))
    wait_final = safe_gate * mad_normalize(wait_score) * 0.24484813561511343 * wait_urgency
    slack_weight = np.clip(1.0 - slack_norm / (slack_norm.max() + eps), 0.0, 1.0)
    rank_weight = 1.0 - slack_weight
    rank_score = safe_gate * (-upward_rank / (np.median(upward_rank) + eps)) * rank_weight * 0.5601638225743357
    score = slack_penalty + critical_release_score
    score += energy_score + energy_uncertainty_score + wait_final + rank_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
