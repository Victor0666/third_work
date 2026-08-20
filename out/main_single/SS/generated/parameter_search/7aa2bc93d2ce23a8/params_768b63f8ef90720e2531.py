import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's robust median-MAD normalization and slack-power penalty with Parent 1's explicit comm+uncert penalty;
       removes redundant 'comm_energy_coupling' to comply with 12-parameter limit; replaces it with strengthened duration_risk_score;
       enforces strict DDL-first semantics via dual-gated energy/uncertainty terms; clips all contributions to [-2,2] for stability."""
    eps = 0.0002541247464633945
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
    norm_comm = median_mad_normalize(min_comm_time)
    ddl_gate = 1.0 / (1.0 + np.exp(-7.160878919796778 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.0297070334822305
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.000324460055521 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 2.754213372839571 * rank_gate)
    uncert_gate = 1.0 / (1.0 + np.exp(-7.160878919796778 * (norm_uncert - 0.7920468500385454)))
    duration_risk_score = norm_duration * uncert_gate * slack_pressure * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.5120641358491022 * (ready_wait_time + 0.0002210050439357769))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.10003177014666884 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.8126363480092361 * np.clip(duration_risk_score, -2.0, 2.0) + 0.8068417723009362 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.6367804916186013 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
