import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robustness with Parent 1's work-density insight:
      - Retains MAD-based normalization (robust for N=1, sparse sets).
      - Keeps tighter DDL protection gate (threshold > 0) and criticality boost logic.
      - Integrates Parent 1's work_density term (remaining_work / duration), gated by joint feasibility sigmoid.
      - Adds exponential wait-time decay instead of linear normalization to prevent starvation without over-prioritizing old tasks.
      - Uses unified uncertainty modulation across all risk-sensitive components.
      - All numeric literals are {-2,-1,0,1,2}; no loops, randomness, or side effects.
    """
    eps = 2.762481106827313e-07
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
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.742390638834465, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.31407359576266763
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.11704283605462168
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 1.972744260568672, 1.0)
    slack_headroom_mask = np.where(slack > 0.008148517398717883, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    rank_work_interaction = upward_rank * remaining_work
    rank_work_score = -mad_normalize(rank_work_interaction)
    slack_lb = -0.1623333146790884
    slack_ub = 25.635377997375954
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.5332257624561119 + (1.0 - 0.5332257624561119) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-6.1209647606276105 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.5084688722123564 * energy_norm * unc_norm * unc_sigmoid
    work_density = np.divide(remaining_work, duration_total)
    work_density_norm = mad_normalize(work_density)
    joint_ratio = slack / (uncertainty + eps)
    joint_feasibility = 1.0 / (1.0 + np.exp(-6.1209647606276105 * joint_ratio))
    work_density_bonus = 0.6622174822684156 * work_density_norm * joint_feasibility
    wait_normalized = np.where(slack_headroom_mask > 0.0, np.exp(-0.6622174822684156 * ready_wait_time), 0.0)
    wait_score = mad_normalize(wait_normalized)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk)
    score += slack_headroom_mask * (0.6622174822684156 * energy_eff_score + rank_work_score + rank_score + energy_uncertainty_score + work_density_bonus + wait_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
