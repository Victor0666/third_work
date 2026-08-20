import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust median-MAD normalization and sharpened gates with Parent 1's validated successor-release interaction.
       Structural novelty: (1) successor-release now couples wait time *and* descendant criticality under DDL breach, (2) unified dual-gated slack-pressure activation,
       (3) strict [-2,2] clipping per term for stability, (4) explicit anti-starvation + anti-lateness decoupling via orthogonal terms."""
    eps = 1.0544894885334976e-06
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
    ddl_gate = 1.0 / (1.0 + np.exp(-2.012319778071909 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.9012019695719933
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure_raw = np.clip(-norm_slack, 0.0, 2.0)
    slack_pressure_gate = 1.0 / (1.0 + np.exp(-6.982657062578684 * (slack_pressure_raw - 1.0 / 2.0)))
    uncert_gate = 1.0 / (1.0 + np.exp(-2.012319778071909 * (norm_uncert - 0.498392445857075)))
    duration_risk_score = norm_duration * uncert_gate * slack_pressure_raw * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.2791132005986809 * (ready_wait_time + 0.00012144524655997419))
    successor_release = norm_wait * norm_rank * ddl_breach
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.9012019695719933 * slack_pressure_raw) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 2.9309009942332005 * slack_pressure_gate), -2.0, 2.0) - 1.6816334052977195 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.6110894079128818 * np.clip(duration_risk_score, -2.0, 2.0) + 0.48328222060770776 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.9012019695719933 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.7126137835969553 * np.clip(norm_work * ddl_gate, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
