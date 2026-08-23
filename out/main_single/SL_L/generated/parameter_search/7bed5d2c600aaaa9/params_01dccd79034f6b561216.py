import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
      - Hard DDL protection gate: explicit slack <= ddl_protection_threshold (not just < 0) using parameterized threshold
      - Unified MAD-based global feature scaling (instead of per-feature percentile) for rank stability across N=1 to large sets
      - Upward-rank × remaining-work interaction unconditionally (confirmed critical-path release)
      - Energy-aware terms gated only under *meaningfully positive slack* (slack > slack_headroom_threshold), not just >0
      - Bounded monotonic gates replacing fragile sigmoids/percentiles; uses tanh and clipped linear
      - Ready-wait-time scaled by inverse slack magnitude, but clamped to avoid explosion near zero
      - All divisions guarded; no infinite/nan outputs; deterministic and finite
    """
    eps = 1.3372843987442153e-09
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
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev) + eps
        return np.clip((x - med) / mad, -2.0, 2.0)
    ddl_protection_threshold = -0.8813866536689545
    is_ddl_risky = slack <= ddl_protection_threshold
    slack_score = np.where(is_ddl_risky, (-slack) ** 3.6952107298483114, 0.0)
    unc_slack_coupling = np.where(is_ddl_risky, uncertainty * (-slack + eps), 0.0) * 0.7671479099816111
    duration_total = min_exec_time + min_comm_time + eps
    duration_risk = duration_total * uncertainty * 1.840335491279973
    critical_path_urgency = upward_rank * remaining_work
    slack_headroom_threshold = -0.8813866536689545 + 1.0
    slack_headroom_mask = np.where(slack > slack_headroom_threshold, 1.0, 0.0)
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    energy_eff_score = mad_normalize(energy_per_work)
    slack_lb = -0.8813866536689545
    slack_ub = 8.558729945221366
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.6262961491853999 + (1.0 - 0.6262961491853999) * (1.0 - slack_scaled)
    rank_score = -np.tanh(mad_normalize(upward_rank)) * weight_rank
    unc_sigmoid = np.tanh(5.502956866216636 * (uncertainty - 1.0))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.2516081795259485 * energy_norm * unc_norm * unc_sigmoid
    wait_base = np.where(slack_headroom_mask > 0.0, ready_wait_time / (np.abs(slack) + 1.0), 0.0)
    wait_score = mad_normalize(wait_base)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_path_urgency)
    score += slack_headroom_mask * (0.8139372768022684 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    critical_gate = np.where(is_ddl_risky, 4.473840109756688, 1.0)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
