import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces unstable power-law coupling with bounded linear-ramp slack interaction;
       reintroduces explicit rank-slack coupling using clipped, sign-preserving ramp (not sigmoid) for deterministic monotonicity;
       eliminates successor_release_score to strictly enforce feasibility-first ranking;
       restores per-term [-2,2] clipping for robust ordinal integrity under distribution shift;
       uses sign-aware robust normalization (median centering preserves directionality)."""
    eps = 0.00028393172738370905
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        if N == 1:
            center = x[0]
            spread = eps
        else:
            center = np.median(x)
            spread = np.median(np.abs(x - center))
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.868261078682067
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_ramp = np.clip((1.0 - norm_slack) * 0.3324906899826371, 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.764911777173567 * slack_ramp)
    ddl_gate = 1.0 / (1.0 + np.exp(-6.1508591115286775 * slack))
    uncert_gate = np.where(norm_uncert > 0.20297364171558008, 1.0, 0.0)
    duration_risk_score = norm_duration * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.04217037862286783 * (norm_wait + 0.0004864329688536066))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    energy_slack_penalty = norm_energy * (1.0 + 1.868261078682067 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.13012317729074663 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.7308396763937703 * np.clip(duration_risk_score, -2.0, 2.0) + 0.5981966422694278 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.868261078682067 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.9636565833555353 * np.clip(norm_work, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
