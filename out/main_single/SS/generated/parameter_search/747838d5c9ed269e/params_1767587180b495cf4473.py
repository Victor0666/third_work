import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: integrates binary DDL breach response with calibrated urgency boost;
       replaces unbounded waiting bonuses with capped logistic relief;
       refines successor-release coupling using slack-gated residual work impact;
       uses N-aware median-MAD normalization with explicit scalar fallback for N=1;
       adds explicit starvation cap to preserve DDL feasibility hierarchy."""
    eps = 0.0003693101168343576
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
    slack_pressure_gate = 1.0 / (1.0 + np.exp(-2.633233900422945 * (slack_gap - eps)))
    residual_work_impact = norm_rank * norm_work * (ddl_breach + (1.0 - ddl_breach) * slack_pressure_gate)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.9745288228149365
    breach_amplified_penalty = raw_slack_penalty * (1.0 + 0.9443216690837734 * ddl_breach)
    norm_slack_penalty = robust_normalize(breach_amplified_penalty)
    boosted_rank = norm_rank * (1.0 + 0.5383895561589447 * slack_pressure_gate)
    uncert_gate = (norm_uncert > 0.17769079621059447).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    duration_risk_score = norm_duration * norm_uncert * slack_pressure_gate
    wait_benefit = np.clip(1.0 - np.exp(-0.22954039010839805 * (norm_wait + 0.0008654802177345767)), 0.0, 0.9443216690837734)
    score = +norm_slack_penalty - residual_work_impact - boosted_rank - 1.5284163338582823 * norm_energy - wait_benefit + 0.7512855809113091 * duration_risk_score + 0.889065662449409 * energy_uncert_penalty + 0.3309228581864537 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
