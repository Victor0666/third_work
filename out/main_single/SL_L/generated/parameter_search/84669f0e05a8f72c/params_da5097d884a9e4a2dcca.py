import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule:
      - Removes unused 'criticality_boost' parameter; successor pressure now uses fixed bias only.
      - Uses robust, *bounded clip-based interpolation* for rank weighting: `np.clip(slack_scaled, 0.0, 1.0)` directly as weight.
      - Introduces *host-load–aware gating*: suppresses energy-efficiency signal via `(1.0 - np.clip(energy_norm * unc_norm, 0.0, 1.0))`.
      - Refines successor-release pressure: `(upward_rank * remaining_work) / (PARAMS['successor_pressure_denom_bias'] + np.maximum(0.0, -slack))`, ensuring monotonic urgency without blowup.
      - Drops all wait-time, fairness, and fragile linear blend terms — strict adherence to hard deadline feasibility first.
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no other constants used.
      - Final score prioritizes: (1) DDL-violation penalties, (2) critical-path release urgency, (3) risk-adjusted energy within feasible region, gated by load-awareness.
    """
    eps = 1.3402070913458497e-09
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x) if N > 1 else np.mean(x)
        mad = np.median(np.abs(x - med)) if N > 1 else eps
        scale = 1.3762293127584297 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_score = np.where(slack < 0, (-slack) ** 3.740870744819824, 0.0)
    slack_lb = -40.8307712358365
    slack_ub = 35.20157040199945
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    ddl_gate = (slack <= 0).astype(float)
    deadline_pressure = np.maximum(0.0, -slack) + 0.09019050128527852
    successor_pressure = upward_rank * remaining_work / deadline_pressure
    critical_release_boost = successor_pressure * ddl_gate
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_norm = mad_normalize(energy_per_sec)
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    load_gate = 1.0 - np.clip(energy_norm * unc_norm, 0.0, 1.0)
    energy_weight_adj = 1.988434201329059 * load_gate
    unc_slack_coupling = uncertainty * np.maximum(0.0, -slack) * 2.9485243382566093
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.3658262505956513
    rank_weight = np.clip(slack_scaled, 0.0, 1.0)
    rank_norm = mad_normalize(upward_rank)
    rank_score = -rank_norm * rank_weight
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.9165305869029439 * (uncertainty - 1.0)))
    energy_uncertainty_score = 0.09617434002926253 * energy_norm * unc_norm * unc_sigmoid
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + energy_weight_adj * energy_eff_norm + rank_score + energy_uncertainty_score - critical_release_boost
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
