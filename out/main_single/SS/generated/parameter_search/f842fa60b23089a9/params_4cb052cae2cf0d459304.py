import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robustness with Parent 1's structured gating:
       - Uses median-MAD normalization (Parent 2) for outlier resilience and sign preservation.
       - Retains smooth ddl_pressure = 1/(1+|slack|+eps) (Parent 2) for stable urgency near zero.
       - Introduces power-law successor_release = (min_exec_time + eps)^p * norm_rank * ddl_breach for stronger amplification of fast blockers.
       - Uses exponential starvation relief: exp(-wait_decay * ready_wait_time) * (1 - ddl_pressure).
       - Adds clipped norm_slack penalty (Parent 2) as primary urgency anchor.
       - Keeps host-load surrogate and energy-uncertainty coupling from Parent 2, but gates them jointly by ddl_pressure AND uncertainty threshold.
       - All composite terms clipped to [-2,2] for bounded AST depth and numerical stability.
    """
    eps = 7.5749465951985435e-06
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
    ddl_pressure = 1.0 / (1.0 + np.abs(slack) + eps)
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_urgency = norm_rank * norm_work * ddl_breach
    successor_release = np.power(min_exec_time + eps, 0.9696774429686008) * norm_rank * ddl_breach
    norm_slack_penalty = np.clip(norm_slack, -2.0, 2.0)
    load_gate = (norm_uncert >= 0.2830193801404228).astype(float) * ddl_pressure
    host_load_surrogate = norm_duration * norm_uncert * load_gate
    energy_uncert_gate = (ddl_pressure > 0.2907722850656488).astype(float) * (norm_uncert >= 0.2830193801404228).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * energy_uncert_gate
    wait_benefit = np.exp(-0.00352936009244079 * ready_wait_time) * (1.0 - ddl_pressure)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_urgency, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - 1.2103392811336922 * np.clip(norm_energy * ddl_pressure, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(host_load_surrogate, -2.0, 2.0) + 0.3683149519578943 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.5735467257026694 * np.clip(norm_work, -2.0, 2.0) + 2.280515528050625 * np.clip(norm_rank * ddl_pressure, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
