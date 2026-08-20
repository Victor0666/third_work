import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule:
       - Restores strict binary deadline-pressure activation for criticality coupling (slack <= 0) — eliminates sigmoid dilution.
       - Replaces unstable pre-clamping with *adaptive outlier suppression via median-based IQR inflation*, preserving rank-order fidelity for N=1.
       - Introduces `urgency_sharpening_factor`: scales MAD-normalized slack to strengthen primary urgency driver without clipping distortion.
       - Removes redundant `successor_release_smooth` term; all urgency is channeled through sharpened slack + conditional critical coupling.
       - All normalization uses safe fallbacks for N=1; no unbounded ops; final score finite, deterministic, shape-(N,)."""
    eps = 0.0005360041776606824
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
            q1 = np.quantile(x, 0.32915401807088823)
            q3 = np.quantile(x, 0.7065529728561425)
        iqr = q3 - q1
        iqr_safe = np.maximum(iqr, eps * (np.abs(q1) + np.abs(q3) + eps))
        return (x - q1) / iqr_safe
    norm_slack = quantile_normalize(slack)
    norm_energy = quantile_normalize(min_incremental_energy)
    norm_duration = quantile_normalize(min_exec_time + min_comm_time)
    norm_rank = quantile_normalize(upward_rank)
    norm_work = quantile_normalize(remaining_work)
    norm_wait = quantile_normalize(ready_wait_time)
    norm_uncert = quantile_normalize(uncertainty)
    slack_med = np.median(slack)
    slack_mad = np.median(np.abs(slack - slack_med))
    slack_spread = np.maximum(slack_mad, eps)
    norm_slack_mad = (slack - slack_med) / slack_spread
    sharpened_urgency = 2.5504184978806945 * norm_slack_mad
    ddl_pressure = (slack <= 0.0).astype(float)
    ddl_feasible_gate = 1.0 / (1.0 + np.exp(-7.008229060961662 * (slack - 0.0)))
    adaptive_uncert_threshold = 0.24817931233981658 * (1.0 - 1.0 / (1.0 + np.exp(-slack)))
    slack_aware_uncert_gate = (uncertainty > adaptive_uncert_threshold).astype(float) * ddl_feasible_gate
    critical_coupling = norm_rank * norm_work * ddl_pressure * 0.9387051021469703
    wait_benefit_pressure = np.clip(0.40474906513933107 * ready_wait_time, 0.0, 2.0) * ddl_pressure
    energy_uncert_coupling = norm_energy * norm_uncert * slack_aware_uncert_gate
    slack_pressure = np.clip(-norm_slack_mad, 0.0, 2.0)
    energy_slack_amplifier = 1.0 + 1.6083844462852046 * slack_pressure
    risk_signal = norm_uncert * ddl_feasible_gate
    score = +np.clip(sharpened_urgency, -2.0, 2.0) - np.clip(critical_coupling, -2.0, 2.0) - np.clip(norm_rank, -2.0, 2.0) - np.clip(wait_benefit_pressure, -2.0, 2.0) - 0.8017725798132576 * np.clip(norm_energy * ddl_feasible_gate, -2.0, 2.0) + 0.2959199633914663 * np.clip(energy_uncert_coupling, -2.0, 2.0) + np.clip(norm_energy * energy_slack_amplifier * ddl_feasible_gate, -2.0, 2.0) + 0.011189188423363498 * np.clip(norm_work * ddl_pressure, -2.0, 2.0) + np.clip(risk_signal, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
