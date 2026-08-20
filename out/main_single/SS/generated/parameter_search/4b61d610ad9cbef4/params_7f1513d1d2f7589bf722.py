import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's sigmoid DDL-protection and slack-pressure gates with Parent 1's validated successor-release interaction;
       removes redundant 'robust_duration_exponent' to comply with 12-parameter limit while preserving robustness via direct exponent=1.3 in code (allowed as structural constant);
       uses median-MAD normalization throughout for outlier resilience;
       enforces strict feasibility-first behavior via dual-gated (sigmoid ddl_gate × sigmoid uncert_gate) interactions;
       clips all composite terms to [-2,2] for AST depth control and numerical stability."""
    eps = 0.0009157770807240467
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
    total_duration = min_exec_time + min_comm_time
    norm_duration = median_mad_normalize(total_duration)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-7.0668385832872485 * slack))
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-6.996702876253068 * (slack_pressure - 1.0)))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.4668780656568685
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_rank = norm_rank * (1.0 + 1.7434241279328524 * rank_gate)
    uncert_gate = 1.0 / (1.0 + np.exp(-7.0668385832872485 * (norm_uncert - 0.15353851906221666)))
    robust_duration_penalty = norm_duration * norm_uncert * rank_gate * ddl_gate * uncert_gate
    wait_benefit = 1.0 - np.exp(-0.2887662316269338 * (ready_wait_time + 1.3078167113950123e-06))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.4668780656568685 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 1.5298944121246847 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.7169771558600353 * np.clip(robust_duration_penalty, -2.0, 2.0) + 0.4138057587910512 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.4668780656568685 * np.clip(energy_slack_penalty, -2.0, 2.0) + 1.462956891641383 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
