import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces fixed zero-slack DDL gate with adaptive threshold gate;
       introduces conditional successor-release interaction for blocked critical paths;
       uses rank-based quantile normalization for robust ordinal preservation under skew;
       removes inactive parameters (duration_robustness, wait_saturation_offset) and reallocates budget."""
    eps = 0.0002933867336296669
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def quantile_normalize(x):
        x = np.copy(x)
        if N == 1:
            return np.zeros_like(x)
        ranks = np.argsort(np.argsort(x))
        percentile = (ranks + 1.0) / (N + 1.0)
        return 2.0 * percentile - 1.0
    norm_slack = quantile_normalize(slack)
    norm_energy = quantile_normalize(min_incremental_energy)
    norm_duration = quantile_normalize(min_exec_time + min_comm_time)
    norm_rank = quantile_normalize(upward_rank)
    norm_work = quantile_normalize(remaining_work)
    norm_wait = quantile_normalize(ready_wait_time)
    norm_uncert = quantile_normalize(uncertainty)
    adaptive_slack = slack - -0.315184373125557
    ddl_gate = 1.0 / (1.0 + np.exp(-2.041541166494044 * adaptive_slack))
    breach_mask = (slack <= 0.0).astype(float)
    release_coupling = np.exp(-0.093601793776443 * remaining_work / (np.abs(slack) + eps))
    successor_release_boost = norm_rank * release_coupling * breach_mask
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.8804061378025225
    norm_slack_penalty = quantile_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 0.8292770706493031 * coupled_slack
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.0285701505419866 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 0.8292770706493031 * rank_gate)
    uncert_gate = 1.0 / (1.0 + np.exp(-2.041541166494044 * (norm_uncert - 0.13215975026062554)))
    duration_risk_score = norm_duration * uncert_gate * slack_pressure * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.39425843195209864 * ready_wait_time)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.8804061378025225 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release_boost, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.8451632630827749 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_risk_score, -2.0, 2.0) + 0.014338123410203478 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.8804061378025225 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.08202822291634994 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
