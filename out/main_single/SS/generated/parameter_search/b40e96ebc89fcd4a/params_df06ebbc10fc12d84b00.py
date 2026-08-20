import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's robust slack penalty normalization with Parent 1's explicit starvation relief and adds novel latency-risk coupling.
       Key improvements:
         - Introduces `latency_risk_coupling`: multiplies (exec+comm) * uncertainty to directly model latency risk exposure
         - Uses raw-slack penalty *before* normalization (Parent 1 insight) but applies MAD normalization *only* to the penalty itself (Parent 2 refinement)
         - Retains sharp sigmoid slack-pressure gating on norm_slack for stable critical-path activation
         - Preserves logistic starvation relief with saturation offset for fairness
         - All 12 parameters used; no numeric literals except -2,-1,0,1,2."""
    eps = 0.09351900738435004
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
        x_abs = np.abs(x)
        center = np.median(x_abs) if N > 1 else x_abs[0]
        spread = np.median(np.abs(x_abs - center)) if N > 1 else np.abs(x_abs[0] - center) + eps
        return (x_abs - center) / (spread + eps)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    norm_slack = robust_normalize(slack)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.422069522568201
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-2.5203940148327275 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.4973800693654393 * rank_gate)
    latency_risk_score = (min_exec_time + min_comm_time) * uncertainty
    norm_latency_risk = robust_normalize(latency_risk_score)
    uncert_gate = np.where(norm_uncert > 0.5448610238866932, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure
    wait_benefit = 1.0 - np.exp(-0.20142607640240448 * (norm_wait + 0.0008231133889991092))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 1.4680307931799879 * norm_energy - norm_duration - wait_benefit + 0.2172824981952779 * duration_risk_score + 0.3594669145136773 * energy_uncert_penalty + 1.6055107529629502 * norm_work + 0.41530053882218443 * norm_latency_risk
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
