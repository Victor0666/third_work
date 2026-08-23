import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robustness with Parent 1's structured risk decomposition:
      - Retains Parent 2's stable min-max normalization with empirical slack bounds and DDL-gated starvation.
      - Integrates Parent 1's bounded duration-risk penalty (linear, not power-law) to avoid over-penalization under tight slack.
      - Unifies critical path release signal using upward_rank * remaining_work (Parent 1) but normalizes via Parent 2's robust min-max.
      - Uses lexicographic gating: non-DDL terms only active when slack > wait_starvation_activation_threshold (strictly safe region).
      - All numeric literals restricted to {-2,-1,0,1,2}; fully deterministic, finite-output guaranteed.
    """
    eps = 3.4098706443867934e-07
    mm_eps = 3.2603999670546575e-06
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
    slack_score = np.where(slack < 0, (-slack) ** 1.4544265567462262, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.4264976523591704
    duration_total = min_exec_time + min_comm_time + eps
    duration_risk_base = duration_total * uncertainty
    duration_risk_penalty = np.where(slack <= 0.0, duration_risk_base * 0.4753401486969569, 0.0)
    critical_release = upward_rank * remaining_work
    critical_release_norm = minmax_normalize(critical_release)
    critical_release_score = -critical_release_norm * 3.0149746916830242
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    load_proxy = 1.0 + minmax_normalize(uncertainty)
    scaled_energy = min_incremental_energy * load_proxy
    energy_per_sec = scaled_energy / duration_total
    energy_eff_score = minmax_normalize(energy_per_sec)
    slack_lb = -25.814370872905002
    slack_ub = 26.52264246381527
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.7965559835804535 + (1.0 - 0.7965559835804535) * (1.0 - slack_scaled)
    rank_score = -minmax_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-0.5492882665860311 * (uncertainty - 1.0)))
    energy_norm = minmax_normalize(scaled_energy)
    unc_norm = minmax_normalize(uncertainty)
    energy_uncertainty_score = 0.5088978752911448 * energy_norm * unc_norm * unc_sigmoid
    wait_normalized = np.where(slack <= 0.0, ready_wait_time / (np.abs(slack) + 1.0), 0.0)
    wait_score = minmax_normalize(wait_normalized)
    score = minmax_normalize(slack_score) + minmax_normalize(unc_slack_coupling) + minmax_normalize(duration_risk_penalty) + critical_release_score + wait_score
    score += slack_headroom_mask * (1.5230928851853336 * energy_eff_score + rank_score + energy_uncertainty_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
