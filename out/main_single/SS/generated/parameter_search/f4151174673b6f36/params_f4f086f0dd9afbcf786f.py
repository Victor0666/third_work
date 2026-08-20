import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust DDL-protection and duration_robustness with Parent 1's validated successor-release coupling.
       Structural novelty: replaces scalar critical_path_leverage with gated, coupling-weighted successor_release term activated only on breach and uncertainty,
       and adds a bounded linear slack-rank interaction that preserves monotonic urgency while avoiding over-parameterization.
       All terms clipped to [-2,2]; median-MAD normalization ensures outlier resilience; final score finite and shape-(N,)."""
    eps = 2.7978946379945902e-05
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
    ddl_gate = 1.0 / (1.0 + np.exp(-7.193215765167416 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    uncert_gate = 1.0 / (1.0 + np.exp(-7.193215765167416 * (norm_uncert - 0.62283433392498)))
    successor_release = norm_rank * norm_work * ddl_breach * uncert_gate
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.7343033991186796
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 0.7436613310081404 * coupled_slack
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.3065967741181344 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 0.7436613310081404 * rank_gate)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.5105224408751082 * (ready_wait_time + 2.477182165751621e-06))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.7343033991186796 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.3336771665219935 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.038580051382337194 * np.clip(duration_risk_score, -2.0, 2.0) + 0.462693112928175 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.7343033991186796 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.2365719542567929 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
