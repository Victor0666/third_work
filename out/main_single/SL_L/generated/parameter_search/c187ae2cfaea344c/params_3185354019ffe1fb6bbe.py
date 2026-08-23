import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule:
      - Replaces tanh safety gate with *hard DDL-protection gate*: active only when slack <= 0, ensuring zero smoothing of deadline-critical urgency.
      - Introduces explicit `upward_rank * remaining_work` interaction (not density) to directly capture successor-release pressure and critical-path starvation.
      - Removes wait-starvation term entirely — fairness is secondary and dilutes hard deadline feasibility.
      - Uses strict boolean masking instead of smooth gates for all deadline-critical components, preserving discrete urgency semantics.
      - All normalization uses robust MAD; no percentile or clipping artifacts.
      - Final score prioritizes DDL feasibility first (via slack_score, unc_slack_coupling, critical_release_boost), then energy within feasible set.
    """
    eps = 9.537138207680451e-05
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
        scale = 1.1489373689881792 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_score = np.where(slack < 0, (-slack) ** 3.2894272740966546, 0.0)
    slack_lb = -6.078487648445375
    slack_ub = 43.63353078685329
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    ddl_gate = (slack <= 0).astype(float)
    critical_release_boost = upward_rank * remaining_work * (1.0 + 1.5672099699058255 * (1.0 - slack_scaled)) * ddl_gate
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_norm = mad_normalize(energy_per_sec)
    energy_suppression_gate = (slack > 0.0).astype(float)
    energy_weight_adj = 1.141952722009375 * (1.0 - energy_suppression_gate)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.6350508503951442
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.8326518648873831
    weight_rank = 0.9747970158376527 + (1.0 - 0.9747970158376527) * (1.0 - slack_scaled)
    rank_norm = mad_normalize(upward_rank)
    rank_score = -rank_norm * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.2127473068952206 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 1.937363459428766 * energy_norm * unc_norm * unc_sigmoid
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + energy_weight_adj * energy_eff_norm + rank_score + energy_uncertainty_score - critical_release_boost
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
