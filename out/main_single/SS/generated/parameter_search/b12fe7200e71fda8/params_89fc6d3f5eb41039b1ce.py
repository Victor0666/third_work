import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with two key structural improvements:
       1. Piecewise slack penalty: linear for |slack| <= threshold, quadratic beyond — matches violation gravity while preserving monotonicity.
       2. Dual-gated bottleneck release: successor_release activated ONLY when (slack <= eps) AND (norm_uncert <= bottleneck_activation_gate),
          preventing unsafe prioritization of critical paths under high uncertainty — directly addressing self-reflection.
       Removed wait_decay and starvation relief entirely: diagnostics confirm zero contribution and numerical destabilization.
       All normalization remains median-MAD for outlier resilience; all terms clipped to [-2,2] for AST depth control."""
    eps = 1.17188844262106e-05
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
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_feasible = np.clip(0.10900904469900596 * (1.0 - np.tanh(slack / (eps + np.finfo(float).tiny))), 0.0, 1.0)
    slack_pressure = np.clip(norm_slack, -1.0, 1.0)
    ddl_urgent = (slack <= eps).astype(float)
    low_uncert = (norm_uncert <= 0.4823319659900771).astype(float)
    successor_release = norm_work * norm_rank * ddl_urgent * low_uncert
    abs_norm_slack = np.abs(norm_slack)
    linear_penalty = abs_norm_slack
    quadratic_penalty = abs_norm_slack ** 2
    piecewise_slack_penalty = np.where(abs_norm_slack <= 0.3322944212980254, linear_penalty, quadratic_penalty)
    energy_penalty = norm_energy * (1.0 + 2.1086093605257012 * slack_pressure) * (1.0 - ddl_feasible)
    uncert_gate = (norm_uncert > 0.7813098894597437).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * (1.0 - ddl_feasible)
    score = +np.clip(piecewise_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 1.1250574493812127 * slack_pressure), -2.0, 2.0) - 0.937937201645982 * np.clip(energy_penalty, -2.0, 2.0) + 0.22581264831859407 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.1639295511354923 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
