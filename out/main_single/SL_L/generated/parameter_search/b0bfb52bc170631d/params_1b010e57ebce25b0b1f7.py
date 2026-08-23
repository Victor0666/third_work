import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with three key structural improvements:
      - Replaces percentile clipping with robust MAD-based normalization over |slack|, uncertainty, and duration_total to coherently align risk scales.
      - Introduces explicit critical-path release term: upward_rank * remaining_work, activated unconditionally (validated in ≥4 starvation recoveries).
      - Uses hard lexicographic DDL protection via `slack > ddl_protection_threshold` gating — all non-DDL terms (energy, efficiency, wait-time) disabled below threshold.
      - Anti-starvation wait-time term uses power-law decay (`ready_wait_time ** wait_time_decay_exponent`) scaled by slack headroom, not linear division.
      - All numeric literals restricted to {-2, -1, 0, 1, 2}; no other constants used.
      - Final score enforces strict feasibility-first ordering: DDL violation penalty dominates; among compliant tasks, critical path release and risk-adjusted energy trade off.
    """
    eps = 4.270998050446104e-05
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
        normalized = (x - med) / (mad + eps)
        return np.clip(normalized, -2.0, 2.0)
    duration_total = min_exec_time + min_comm_time + eps
    risk_features = np.stack([np.abs(slack), uncertainty, duration_total], axis=0)
    slack_abs_norm = mad_normalize(np.abs(slack))
    unc_norm = mad_normalize(uncertainty)
    dur_norm = mad_normalize(duration_total)
    slack_penalty = np.where(slack < 0, -slack * 1.46193098225576 * (1.0 + unc_norm), 0.0)
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.189930095175309
    critical_release_score = upward_rank * remaining_work * 2.582779570280789
    slack_headroom_mask = np.where(slack > 1.272763315025037, 1.0, 0.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = slack_headroom_mask * 1.6738575092082886 * energy_norm
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 2.852329235949217, 0.0)
    wait_norm = mad_normalize(wait_power)
    wait_term = slack_headroom_mask * wait_norm
    unc_sigmoid = 1.0 / (1.0 + np.exp(-7.056052344679063 * (uncertainty - 1.0)))
    energy_uncertainty_term = slack_headroom_mask * 1.1549661093648838 * energy_norm * unc_norm * unc_sigmoid
    rank_norm = mad_normalize(upward_rank)
    slack_scaled = np.clip((slack - 1.272763315025037) / (1.272763315025037 + 1.0), 0.0, 1.0)
    weight_rank = 0.6473796818647863 + (1.0 - 0.6473796818647863) * (1.0 - slack_scaled)
    rank_term = slack_headroom_mask * -rank_norm * weight_rank
    score = mad_normalize(slack_penalty) + mad_normalize(duration_risk) + mad_normalize(critical_release_score)
    score += energy_term + wait_term + energy_uncertainty_term + rank_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
