import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining best practices:
       - Uses tunable quantile normalization (Parent 2) for robustness.
       - Applies sign-preserving MAD-normalized slack (Parent 2) for sharp deadline violation sensitivity.
       - Introduces dual-gated criticality coupling: upward_rank × remaining_work activated only under both slack <= 0 AND high congestion.
       - Adds congestion-suppressed starvation relief: linear wait_benefit scaled by (1 - congestion_gate).
       - Keeps DDL-feasibility gating for energy terms (Parent 2) but adds congestion-thresholded activation.
       - All outputs bounded, finite, deterministic, and shape-(N,)."""
    eps = 1.6630138982850656e-06
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
            q1 = q3 = x[0]
        else:
            q1 = np.quantile(x, 0.2899273610259143)
            q3 = np.quantile(x, 0.8117983241376915)
        iqr = q3 - q1 if q3 > q1 else eps
        return (x - q1) / iqr
    norm_slack = quantile_normalize(slack)
    norm_energy = quantile_normalize(min_incremental_energy)
    norm_duration = quantile_normalize(min_exec_time + min_comm_time)
    norm_rank = quantile_normalize(upward_rank)
    norm_work = quantile_normalize(remaining_work)
    norm_wait = quantile_normalize(ready_wait_time)
    norm_uncert = quantile_normalize(uncertainty)
    slack_med = np.median(slack)
    slack_mad = np.median(np.abs(slack - slack_med))
    slack_spread = slack_mad if slack_mad > eps else eps
    norm_slack_mad = (slack - slack_med) / slack_spread
    ddl_feasible = (slack >= 0.0).astype(float)
    ddl_feasible_gate = 1.0 / (1.0 + np.exp(-3.446272820595674 * (slack - 0.0)))
    ddl_pressure = (slack <= 0.0).astype(float)
    congestion = ready_wait_time + uncertainty
    congestion_norm = quantile_normalize(congestion)
    congestion_gate = 1.0 / (1.0 + np.exp(-3.446272820595674 * (congestion_norm - 0.0006539624496669707)))
    dual_critical_gate = ddl_pressure * congestion_gate
    critical_coupling = norm_rank * norm_work * dual_critical_gate * 1.4761819415712951
    wait_benefit = np.clip(0.3488472993376644 * ready_wait_time, 0.0, 2.0)
    suppressed_wait = wait_benefit * (1.0 - congestion_gate) * 0.9703330646602767
    energy_uncert_coupling = norm_energy * norm_uncert * congestion_gate * ddl_feasible
    slack_pressure = np.clip(-norm_slack_mad, 0.0, 2.0)
    energy_slack_amplifier = 1.0 + 1.5395956789381546 * slack_pressure
    score = +np.clip(norm_slack_mad, -2.0, 2.0) - np.clip(critical_coupling, -2.0, 2.0) - np.clip(norm_rank, -2.0, 2.0) - np.clip(suppressed_wait, -2.0, 2.0) - 0.41863855751299317 * np.clip(norm_energy * ddl_feasible_gate, -2.0, 2.0) + 0.2430971511169993 * np.clip(energy_uncert_coupling, -2.0, 2.0) + np.clip(norm_energy * energy_slack_amplifier * ddl_feasible, -2.0, 2.0) + 0.9494383377449096 * np.clip(norm_work * ddl_pressure, -2.0, 2.0) + np.clip(congestion_norm * ddl_pressure, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
