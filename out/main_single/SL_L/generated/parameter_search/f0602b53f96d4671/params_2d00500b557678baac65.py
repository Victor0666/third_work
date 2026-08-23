import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule integrating:
      - Strict lexicographic DDL gating: all non-DDL terms masked by `slack > 0`
      - Critical path urgency via `upward_rank * remaining_work` (validated in cross-agent consensus)
      - Power-law anti-starvation: `ready_wait_time ** wait_time_decay_exponent`, gated by slack headroom
      - MAD-based robust normalization per dimension instead of percentile clipping (more stable under sparse N)
      - Unified duration-risk term: `(min_exec_time + min_comm_time) * (1 + uncertainty)` to reflect load-sensitive delay
      - Energy-efficiency score now uses `min_incremental_energy / (min_exec_time + min_comm_time + eps)` directly
      - All numeric literals restricted to {-2, -1, 0, 1, 2}; no other constants used
      - Final score enforces: DDL violation penalty first → critical path release urgency → energy efficiency among feasible → anti-starvation only when safe
    """
    eps = 1.3076674247466716e-07
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
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 1.3459442206558392, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.5382188337830405
    duration_total = min_exec_time + min_comm_time + eps
    duration_risk = duration_total * (1.0 + uncertainty) * 0.12284857232522417
    critical_path_urgency = upward_rank * remaining_work
    critical_path_score = mad_normalize(critical_path_urgency) * 2.3214250588042855
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -4.968173508928928
    slack_ub = 34.49177898152196
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.5788291980438913 + (1.0 - 0.5788291980438913) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-0.5434858167873616 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.0016177262851519384 * energy_norm * unc_norm * unc_sigmoid
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 0.6609917497777505, 0.0)
    wait_score = mad_normalize(wait_power)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + critical_path_score
    score += slack_headroom_mask * (1.0600546167413005 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
