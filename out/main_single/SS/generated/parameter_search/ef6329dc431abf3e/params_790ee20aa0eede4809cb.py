import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces unstable power-law rank-slack coupling with *clipped linear hinge* for robust monotonic urgency;
       unifies all risk terms under a single *joint slack-uncertainty gate* to reduce over-parameterization;
       introduces *normalized wait-time saturation* using softplus instead of exp to prevent overflow;
       removes redundant duration_robustness — now fully absorbed into joint_gate and slack_pressure scaling;
       retains validated critical_path_leverage and DDL-protection semantics while tightening AST depth and eliminating inf-producing branches."""
    eps = 0.029731795072836274
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def median_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    joint_gate = 1.0 / (1.0 + np.exp(-6.156430894964114 * slack)) * 1.0 / (1.0 + np.exp(6.156430894964114 * (norm_uncert - 0.46104827278707733)))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.021879232656165
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_hinge = np.clip(-norm_slack, 0.0, 1.0)
    rank_slack_coupling = 1.0 + 1.2512613895332578 * slack_hinge
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.6501709906022772 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 1.2512613895332578 * rank_gate)
    wait_benefit = np.log1p(np.exp(-0.438868486996936 * (ready_wait_time + eps)))
    duration_risk_score = norm_duration * norm_uncert * joint_gate
    energy_uncert_penalty = norm_energy * norm_uncert * joint_gate
    energy_slack_penalty = norm_energy * slack_pressure * joint_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.377860862576079 * np.clip(norm_energy * joint_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.34062082449023307 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.021879232656165 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.7907566746553572 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
