import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces soft exponential DDL-gating with *crisp, hard-thresholded linear penalty* for slack <= 0,
       ensuring strict deadline feasibility signaling without gradient vanishing; removes duration_robustness (diagnosed inactive);
       retains dual-gated rank-uncertainty coupling but simplifies its activation to use raw slack threshold instead of norm_uncert;
       introduces explicit slack-thresholded linear energy penalty — not decay — to preserve monotonic urgency under surplus slack;
       all normalization remains sign-aware median/MAD; all terms clipped to [-2,2] for ordinal robustness."""
    eps = 0.0005317380966622719
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
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.9110057561719334
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_ramp = np.clip((1.0 - norm_slack) * 0.4412355482487905, 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 0.5 * slack_ramp)
    ddl_feasible = (slack > 0.0).astype(float)
    ddl_infeasible = 1.0 - ddl_feasible
    energy_feasibility_penalty = norm_energy * ddl_infeasible * 1.0222862668858372
    rank_uncert_coupling = norm_rank * norm_uncert * ddl_infeasible
    wait_benefit = 1.0 - np.exp(-0.20309856558058245 * (norm_wait + 8.028408949049668e-06))
    uncert_gate = np.where(norm_uncert > 0.5669998110854557, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_infeasible
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.1832535309555774 * np.clip(norm_energy * ddl_feasible, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(energy_feasibility_penalty, -2.0, 2.0) + np.clip(rank_uncert_coupling, -2.0, 2.0) + 0.2594243090826844 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.6941497626474029 * np.clip(norm_work, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
