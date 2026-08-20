import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: integrates binary DDL breach response with calibrated urgency boost;
       replaces unbounded waiting bonuses with capped logistic relief;
       refines successor-release coupling using slack-gated residual work impact;
       uses N-aware median-MAD normalization with explicit scalar fallback for N=1;
       adds explicit starvation cap to preserve DDL feasibility hierarchy."""
    eps = 0.0636430407707215
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
            spread = np.abs(x[0]) + eps
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
    ddl_breach = (slack <= 0.0).astype(float)
    slack_gap = np.maximum(-slack, 0.0)
    slack_pressure_gate = 1.0 / (1.0 + np.exp(-3.7252196544879776 * (slack_gap - eps)))
    residual_work_impact = norm_rank * norm_work * (ddl_breach + (1.0 - ddl_breach) * slack_pressure_gate)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.306367459421503
    breach_amplified_penalty = raw_slack_penalty * (1.0 + 0.976419967674948 * ddl_breach)
    norm_slack_penalty = robust_normalize(breach_amplified_penalty)
    boosted_rank = norm_rank * (1.0 + 2.297665531879921 * slack_pressure_gate)
    uncert_gate = (norm_uncert > 0.9892618231168682).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    duration_risk_score = norm_duration * norm_uncert * slack_pressure_gate
    wait_benefit = np.clip(1.0 - np.exp(-0.0005201804014727589 * (norm_wait + 0.00015437851105479334)), 0.0, 0.976419967674948)
    score = +norm_slack_penalty - residual_work_impact - boosted_rank - 0.7152381204507559 * norm_energy - wait_benefit + 1.094090926666142 * duration_risk_score + 0.008180704484988534 * energy_uncert_penalty + 0.3323567036275161 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
