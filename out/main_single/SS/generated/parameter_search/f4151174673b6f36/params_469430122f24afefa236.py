import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust DDL-protection and duration_robustness with Parent 1's validated successor-release coupling.
       Structural novelty: replaces scalar critical_path_leverage with gated, coupling-weighted successor_release term activated only on breach and uncertainty,
       and adds a bounded linear slack-rank interaction that preserves monotonic urgency while avoiding over-parameterization.
       All terms clipped to [-2,2]; median-MAD normalization ensures outlier resilience; final score finite and shape-(N,)."""
    eps = 9.478364040079955e-05
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
    ddl_gate = 1.0 / (1.0 + np.exp(-6.503697156593238 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    uncert_gate = 1.0 / (1.0 + np.exp(-6.503697156593238 * (norm_uncert - 0.933027085850196)))
    successor_release = norm_rank * norm_work * ddl_breach * uncert_gate
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.2074991610466985
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 1.3983852548751963 * coupled_slack
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.900308491469775 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 1.3983852548751963 * rank_gate)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.14708251258804167 * (ready_wait_time + 1.5590894992856917e-06))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 2.2074991610466985 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.5279065562627008 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.8491488206239916 * np.clip(duration_risk_score, -2.0, 2.0) + 0.3910582878708743 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 2.2074991610466985 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.0026554738377848495 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
