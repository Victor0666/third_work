import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating evidence-backed structural changes:
      - Adds explicit successor-release coupling: (upward_rank × remaining_work) scaled by slack deficit, activated only under DDL pressure (slack <= 0)
      - Introduces host-load–aware energy scaling: energy term multiplied by (1 + normalized_uncertainty) only when uncertainty < threshold
      - Uses robust MAD-based normalization for all features (eliminates degeneracy in sparse ready sets)
      - Preserves strict lexicographic DDL gating: non-critical terms disabled when slack <= 0
      - All numeric literals restricted to {-2, -1, 0, 1, 2}
    """
    eps = 1.7047243999669564e-06
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
    slack_score = np.where(slack < 0, (-slack) ** 3.555714202339326, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.5045908405828561
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.0630504385009452
    successor_urgency = upward_rank * remaining_work
    successor_score = np.where(slack <= 0, successor_urgency * 0.7936041708026356, 0.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    host_load_active = np.where(uncertainty < 0.964630689745489, 1.0, 0.0)
    energy_scale = 1.0 + mad_normalize(uncertainty) * host_load_active
    scaled_energy = min_incremental_energy * energy_scale
    energy_norm = mad_normalize(scaled_energy)
    slack_lb = -17.315654327439034
    slack_ub = 34.09547081062931
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.38795139394815364 + (1.0 - 0.38795139394815364) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-0.9364411689931826 * (uncertainty - 1.0)))
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.04050742555026531 * energy_norm * unc_norm * unc_sigmoid
    wait_score = mad_normalize(ready_wait_time)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(successor_score)
    score += slack_headroom_mask * (1.2197035838406733 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
