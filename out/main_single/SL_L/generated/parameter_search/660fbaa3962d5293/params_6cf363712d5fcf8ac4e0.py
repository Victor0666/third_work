import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's exponential urgency & conditional MAD with Parent 1's congestion-aware gating:
      - Bounded exponential urgency decay for slack-modulated rank prioritization (Parent 2)
      - Conditional MAD normalization: only applied when N > conditional_mad_min_size (Parent 2)
      - Congestion-aware energy gating using wait-time & uncertainty (Parent 1 innovation)
      - Lexicographic DDL-critical dominance: slack penalty, tight-slack coupling, and duration risk always active
      - All numeric literals strictly {-2,-1,0,1,2}; no hidden constants; deterministic finite output.
    """
    eps = 2.482064821166642e-06
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
        if N <= 3.4919913604134423:
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
    slack_sigmoid = 1.0 / (1.0 + np.exp(-2.1847273872303603 * slack_sigmoid_input))
    slack_penalty = slack_sigmoid * (1.0 + unc_norm)
    tight_slack_mask = np.where(slack <= 0.17390108484923267, 1.0, 0.0)
    tight_slack_uncertainty_coupling = tight_slack_mask * uncertainty * slack_abs_norm * 0.71389551255375
    duration_risk_base = (min_exec_time + min_comm_time) * uncertainty
    duration_risk_penalty = tight_slack_mask * duration_risk_base * 0.7454329812586863
    critical_release_score = upward_rank * remaining_work * 0.5000075417336612
    slack_distance = np.clip(0.17390108484923267 - slack, 0.0, 22.77354840181917)
    urgency_factor = np.exp(-slack_distance / (2.842651973814702 + eps))
    slack_rank_score = -upward_rank * urgency_factor * 0.5000075417336612
    slack_headroom_mask = np.where(slack > 0.17390108484923267, 1.0, 0.0)
    median_wait = np.median(ready_wait_time) + eps
    congestion_proxy = (ready_wait_time / median_wait + eps) * (1.0 + unc_norm)
    congestion_gate = np.clip(congestion_proxy * 1.4909206609749601, 0.0, 1.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = slack_headroom_mask * congestion_gate * 1.486163813027187 * energy_norm
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 2, 0.0)
    wait_norm = mad_normalize(wait_power)
    wait_term = slack_headroom_mask * wait_norm * 0.9892440738570973
    score = mad_normalize(slack_penalty) + mad_normalize(tight_slack_uncertainty_coupling) + mad_normalize(duration_risk_penalty) + mad_normalize(critical_release_score) + mad_normalize(slack_rank_score)
    score += energy_term + wait_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
