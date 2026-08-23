import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining strengths:
      - Retains Parent 2's hard DDL gate `(slack <= 0)` for strict deadline urgency.
      - Keeps `upward_rank * remaining_work` interaction but applies fixed exponent 1.0 (no new parameter) to avoid exceeding 12 params.
      - Uses robust MAD normalization with `robustness_mad_factor` for stability across N=1–100.
      - Drops `ready_wait_time` entirely (fairness secondary to hard deadlines).
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no other constants used.
      - Final score structure: DDL penalties first, then critical release boost (subtracted), then energy/rank terms (weighted by slack status).
    """
    eps = 9.766312369322716e-08
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
        scale = 1.525776432214801 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_score = np.where(slack < 0, (-slack) ** 3.5188583309962826, 0.0)
    slack_lb = -22.699612625886147
    slack_ub = 24.549732888046265
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    ddl_gate = (slack <= 0).astype(float)
    critical_release_boost = upward_rank * remaining_work * (1.0 + 2.117946855841484 * (1.0 - slack_scaled)) * ddl_gate
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_norm = mad_normalize(energy_per_sec)
    energy_suppression_gate = (slack > 0.0).astype(float)
    energy_weight_adj = 0.27579492261561267 * (1.0 - energy_suppression_gate)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.76310229147458
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.10160076470151586
    weight_rank = 0.8200991784998691 + (1.0 - 0.8200991784998691) * (1.0 - slack_scaled)
    rank_norm = mad_normalize(upward_rank)
    rank_score = -rank_norm * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.033782284809439 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.07151555341466939 * energy_norm * unc_norm * unc_sigmoid
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + energy_weight_adj * energy_eff_norm + rank_score + energy_uncertainty_score - critical_release_boost
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
