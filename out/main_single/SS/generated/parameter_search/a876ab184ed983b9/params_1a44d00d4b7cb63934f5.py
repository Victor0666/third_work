import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's stability with Parent 1's structured coupling:
       - Uses tanh-based DDL feasibility gate (Parent 2) for monotonic, bounded transitions
       - Retains direct successor_release = norm_work * norm_rank * (slack <= eps) (Parent 2's validated core)
       - Adds weighted successor_release term with tunable amplification (new structural parameter)
       - Keeps median-MAD normalization for outlier resilience
       - Replaces unstable sigmoid gates entirely; avoids all unbounded exponentials
       - Enforces strict clipping [-2,2] on all interaction terms for AST depth control
       - Eliminates redundant slack_margin_ratio_threshold and ddl_protection_gate_slope per diagnostics
       - Introduces explicit weight for successor-release coupling to allow CMA-ES tuning of bottleneck emphasis"""
    eps = 4.6471822765121444e-05
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
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_feasible = np.clip(0.7247727609965722 * (1.0 - np.tanh(slack / (eps + np.finfo(float).tiny))), 0.0, 1.0)
    slack_pressure = np.clip(norm_slack, -1.0, 1.0)
    ddl_urgent = (slack <= eps).astype(float)
    successor_release = norm_work * norm_rank * ddl_urgent
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.1802522966297273
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    energy_penalty = norm_energy * (1.0 + 1.1802522966297273 * slack_pressure) * (1.0 - ddl_feasible)
    uncert_gate = (norm_uncert > 0.4870741574430471).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * (1.0 - ddl_feasible)
    wait_benefit = 1.0 - np.exp(-0.23826857412849636 * ready_wait_time)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - 1.7860650701417744 * np.clip(successor_release, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 1.7392730953256474 * slack_pressure), -2.0, 2.0) - 0.37650109780283847 * np.clip(energy_penalty, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.43468282087177096 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.5447680547989258 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
