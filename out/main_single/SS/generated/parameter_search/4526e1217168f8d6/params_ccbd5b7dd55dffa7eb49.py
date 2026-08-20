import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's DDL-protection gate and robust normalization with Parent 1's smooth energy ramp;
       introduces *dual-gated energy activation*: both DDL-feasibility (smooth Heaviside) AND marginal slack alignment (linear ramp);
       replaces brittle critical-path leverage with *normalized successor-unblocking potential* — product of upward_rank and normalized remaining_work,
       gated by slack pressure only (not just breach), improving upstream prioritization under tight-but-feasible deadlines;
       uses median-MAD normalization throughout for outlier resilience;
       all terms clipped to [-2,2] for stability and bounded AST depth."""
    eps = 0.0002898144208174345
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
    ddl_gate = 1.0 / (1.0 + np.exp(-3.3166699398115598 * slack))
    energy_ramp_center = 0.0
    energy_ramp_width = 1.0 / (3.3166699398115598 + eps)
    energy_ramp = np.clip((slack - energy_ramp_center) / (energy_ramp_width + eps), 0.0, 1.0)
    energy_activation = ddl_gate * energy_ramp
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.3832170543977251 * (slack_pressure - 1.0)))
    unblocking_potential = norm_rank * norm_work * rank_gate
    uncert_gate = 1.0 / (1.0 + np.exp(-3.3166699398115598 * (norm_uncert - 0.3344095341338833)))
    duration_risk_score = norm_duration * norm_uncert * rank_gate * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.06275035204980434 * (ready_wait_time + 1.0650881384985817e-07))
    energy_uncert_penalty = norm_energy * norm_uncert * energy_activation
    energy_slack_penalty = norm_energy * (1.0 + 1.7188554973925587 * slack_pressure) * energy_activation
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.7188554973925587
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(unblocking_potential, -2.0, 2.0) - np.clip(0.982627041248669 * norm_energy * energy_activation, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.6511209098723056 * np.clip(duration_risk_score, -2.0, 2.0) + 0.8233128272792961 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.7188554973925587 * np.clip(energy_slack_penalty, -2.0, 2.0) + 1.344430539440078 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
