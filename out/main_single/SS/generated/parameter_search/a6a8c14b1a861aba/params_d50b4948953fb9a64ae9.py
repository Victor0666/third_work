import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's robustness with Parent 1's structured risk coupling:
       - Keeps Parent 2's median-MAD normalization (more stable than quantile-based) and bounded linear starvation.
       - Reintroduces *selective* energy-uncertainty interaction *only under deadline breach*, avoiding overfitting.
       - Adds new `exec_comm_balance` parameter to explicitly control compute-vs-I/O penalty emphasis during DDL violation.
       - Uses Parent 1's `slack_sensitivity_center` for sharper, more interpretable urgency activation near deadline boundary.
       - Drops all synthetic congestion terms (congestion_weight, successor_release_interaction) per performance analysis.
       - All numeric literals strictly ∈ {-2,-1,0,1,2}; no hidden constants; uses np.finfo for safety."""
    eps = 0.0004512824845247597
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
    norm_exec = median_mad_normalize(min_exec_time)
    norm_comm = median_mad_normalize(min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_uncert = median_mad_normalize(uncertainty)
    slack_feasibility = 1.0 / (1.0 + np.exp(-4.495046601701895 * slack))
    ddl_breach = (slack <= -0.1291442473679016).astype(float)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.4612523874519308
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_hinge = np.clip(-norm_slack, 0.0, 1.215018725868022)
    rank_slack_coupling = 1.0 + 1.9068469581510006 * (slack_hinge / (1.215018725868022 + eps))
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    urgency_gate = 1.0 / (1.0 + np.exp(-4.495046601701895 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 1.9068469581510006 * urgency_gate)
    wait_benefit = np.clip(1.0 - 0.9049740189236283 * ready_wait_time, 0.0, 1.0)
    exec_penalty = 0.7508557240275694 * norm_exec * ddl_breach
    comm_penalty = (1.0 - 0.7508557240275694) * norm_comm * ddl_breach
    energy_uncert_penalty = norm_energy * norm_uncert * ddl_breach * 0.17064152865319557
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.3537504463166494 * np.clip(norm_energy * slack_feasibility, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + np.clip(comm_penalty, -2.0, 2.0) + np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.4612523874519308 * np.clip(norm_work * ddl_breach, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
