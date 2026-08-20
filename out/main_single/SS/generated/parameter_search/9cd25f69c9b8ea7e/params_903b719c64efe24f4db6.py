import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's smooth urgency gate and rank floor with Parent 1's successor-release interaction.
       Uses robust median-MAD normalization; all operations safeguarded against NaN/inf/zero.
       Prioritizes hard DDL feasibility first, then critical path, then energy efficiency, then starvation relief — with strict ordering via gated terms."""
    eps = 0.0006847845868286291
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
        x = np.asarray(x, dtype=float)
        center = np.median(x) if N > 1 else x[0]
        spread = np.median(np.abs(x - center)) if N > 1 else np.abs(x[0] - center) + eps
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    median_abs_slack = np.median(np.abs(slack)) if N > 1 else np.abs(slack[0])
    urgency_input = -slack / (median_abs_slack + eps)
    ddl_urgency_gate = 1.0 / (1.0 + np.exp(-1.850113442070852 * urgency_input))
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.9748396327739104
    rank_boost = 0.01842008729891525 + (1.0 - 0.01842008729891525) * ddl_urgency_gate
    protected_rank = norm_rank * rank_boost
    successor_release_impact = norm_rank * norm_work
    wait_raw = 0.4579801294930655 * (ready_wait_time + 7.060522976988001e-08)
    wait_benefit = np.clip(1.0 - np.exp(-wait_raw), 0.0, 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * np.where((norm_energy > 0.0) & (norm_uncert > 0.4711272320766276), 1.0, 0.0)
    duration_risk_score = norm_duration * np.where((slack <= eps) & (uncertainty >= 0.4711272320766276), 1.0, 0.0)
    score = +raw_slack_penalty - 0.5195148287571653 * protected_rank - successor_release_impact - 1.9954262494453452 * norm_energy - norm_duration - wait_benefit + 0.3243667216205812 * duration_risk_score + 0.5849500830662412 * energy_uncert_penalty + 1.5331542741760678 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
