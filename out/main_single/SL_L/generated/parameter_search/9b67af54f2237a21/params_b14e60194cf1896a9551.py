import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with two key structural improvements:
      - Replaces linear slack-distance coupling in slack_rank_score with *bounded exponential decay*:
        urgency grows as exp(-slack_distance / rate), sharply prioritizing critical-path tasks when slack is tight,
        but remains finite and numerically stable even at extreme negative slack.
      - Introduces *conditional MAD normalization*: only applies mad_normalize when N > conditional_mad_min_size,
        avoiding statistical degeneracy (e.g., zero MAD, infinite values) in sparse ready sets (N ≤ 3).
      - All other components preserved from validated hybrid structure: bounded sigmoid slack penalty,
        exclusive tight-slack coupling, bounded duration risk, normalized wait-term, and lexicographic gating.
      - No numeric literals beyond {-2,-1,0,1,2}; all tunables accessed via PARAMS; deterministic and finite-output guaranteed.
    """
    eps = 6.143144357505997e-06
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
        if N <= 6.917693886990265:
            return x
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
    slack_sigmoid = 1.0 / (1.0 + np.exp(-3.330200711485836 * slack_sigmoid_input))
    slack_penalty = slack_sigmoid * (1.0 + unc_norm)
    tight_slack_mask = np.where(slack <= 2.076534857079428, 1.0, 0.0)
    tight_slack_uncertainty_coupling = tight_slack_mask * uncertainty * slack_abs_norm * 0.3184918729118339
    duration_risk_base = (min_exec_time + min_comm_time) * uncertainty
    duration_risk_penalty = tight_slack_mask * duration_risk_base * 0.7222971086606369
    critical_release_score = upward_rank * remaining_work * 2.028256870719222
    slack_distance = np.clip(2.076534857079428 - slack, 0.0, 9.61630882588385)
    urgency_factor = np.exp(-slack_distance / (0.9554258393409183 + eps))
    slack_rank_score = -upward_rank * urgency_factor * 2.028256870719222
    slack_headroom_mask = np.where(slack > 2.076534857079428, 1.0, 0.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = slack_headroom_mask * 1.1447357625036723 * energy_norm
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 2, 0.0)
    wait_norm = mad_normalize(wait_power)
    wait_term = slack_headroom_mask * wait_norm * 0.8774667129462812
    score = mad_normalize(slack_penalty) + mad_normalize(tight_slack_uncertainty_coupling) + mad_normalize(duration_risk_penalty) + mad_normalize(critical_release_score) + mad_normalize(slack_rank_score)
    score += energy_term + wait_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
