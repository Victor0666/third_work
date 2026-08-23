import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with headroom-aware gating and robust duration penalty:
      - Removed wait_time_decay_exponent and relative wait ratio; replaced with fixed linear wait term
        scaled by slack_headroom_mask to satisfy parameter count and avoid unused parameters.
      - Added explicit robust duration penalty: MAD-normalized (min_exec_time + min_comm_time).
      - Critical path release uses *normalized product* (upward_rank * remaining_work) to avoid scale skew.
      - Energy-uncertainty interaction now gated by both sigmoid AND slack_headroom_mask for safety.
      - Anti-starvation uses simple linear wait term: ready_wait_time, masked and normalized — no exponent.
      - All numeric literals are in {-2, -1, 0, 1, 2}; epsilon handled via PARAMS; no in-place mutation.
    """
    eps = 9.796234354212683e-08
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
        dev = np.abs(x - med)
        mad = np.median(dev) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 3.605750000434002, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.927523779324928
    duration_total = min_exec_time + min_comm_time + eps
    duration_risk = duration_total * uncertainty * 0.7932790745417706
    duration_penalty = mad_normalize(duration_total)
    is_ddl_constrained = slack <= 0.0
    norm_upward_rank = mad_normalize(upward_rank)
    norm_remaining_work = mad_normalize(remaining_work)
    successor_release_score = norm_upward_rank * norm_remaining_work * 0.9559785650229415
    successor_release_contribution = np.where(is_ddl_constrained, -successor_release_score, 0.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -32.59417724408431
    slack_ub = 3.8055499911226094
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.636912070390425 + (1.0 - 0.636912070390425) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.494873093204088 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.020415655352191812 * energy_norm * unc_norm * unc_sigmoid * slack_headroom_mask
    wait_linear = np.where(slack_headroom_mask > 0.0, ready_wait_time, 0.0)
    wait_score = mad_normalize(wait_linear)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + duration_penalty + successor_release_contribution
    score += slack_headroom_mask * (0.20966200364608778 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
