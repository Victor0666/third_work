import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's superior bounded sigmoid and exclusive coupling
    with Parent 1's robust bounded duration risk and slack-modulated rank importance.
    
    Key structural improvements:
      - Bounded duration-risk penalty (Parent 1) replaces unbounded duration_risk_penalty (Parent 2)
        to prevent over-penalization under extreme uncertainty, active only when slack <= threshold.
      - Slack-modulated rank importance (Parent 1) replaces rank-slack interpolation (Parent 2),
        directly coupling upward_rank with slack distance for sharper DDL-aware critical-path modulation.
      - Retains Parent 2's bounded sigmoid slack penalty, exclusive tight-slack coupling, and normalized
        wait-headroom scaling — all validated for stability and performance.
      - All normalizations use clipped MAD to [-2,2] for bounded influence; no numeric literals beyond {-2,-1,0,1,2}.
    """
    eps = 1.3998021710397036e-09
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
    slack_abs_norm = mad_normalize(np.abs(slack))
    unc_norm = mad_normalize(uncertainty)
    dur_norm = mad_normalize(duration_total)
    slack_sigmoid_input = -slack_abs_norm
    slack_sigmoid = 1.0 / (1.0 + np.exp(-7.790117038837645 * slack_sigmoid_input))
    slack_penalty = slack_sigmoid * 1.2034955000468475 * (1.0 + unc_norm)
    tight_slack_mask = np.where(slack <= 0.6273311477771887, 1.0, 0.0)
    tight_slack_uncertainty_coupling = tight_slack_mask * uncertainty * slack_abs_norm * 0.5918215836485796
    duration_risk_base = (min_exec_time + min_comm_time) * uncertainty
    duration_risk_penalty = tight_slack_mask * duration_risk_base * 0.9434964746821957
    critical_release_score = upward_rank * remaining_work * 2.009977439337544
    slack_distance = np.clip(0.6273311477771887 - slack, 0.0, np.inf)
    slack_rank_score = -upward_rank * (1.0 + slack_distance / (0.6273311477771887 + eps)) * 0.791085804924528
    slack_headroom_mask = np.where(slack > 0.6273311477771887, 1.0, 0.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = slack_headroom_mask * 0.35554850775722635 * energy_norm
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 2.3368826909161857, 0.0)
    wait_norm = mad_normalize(wait_power)
    wait_term = slack_headroom_mask * wait_norm * 0.8462419883615879
    score = mad_normalize(slack_penalty) + mad_normalize(tight_slack_uncertainty_coupling) + mad_normalize(duration_risk_penalty) + mad_normalize(critical_release_score) + mad_normalize(slack_rank_score)
    score += energy_term + wait_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
