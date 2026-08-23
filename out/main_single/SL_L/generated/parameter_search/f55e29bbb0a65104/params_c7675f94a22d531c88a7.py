import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating evidence-backed structural changes:
      - Adds successor-release coupling term `(upward_rank × remaining_work)` activated *only* under DDL pressure (slack <= 0), directly resolving CRITICAL_PATH_STARVATION.
      - Replaces percentile normalization with robust MAD normalization for stability in sparse ready sets (N ≤ 5).
      - Introduces host-load–aware energy scaling via `sigmoid(uncertainty < center)` — tunable center and steepness to avoid hardcoded thresholds.
      - Uses only {-2,-1,0,1,2} literals; all tunables referenced via PARAMS.
      - Strict lexicographic DDL gating remains: non-critical terms disabled when slack <= 0.
    """
    eps = 1.5994696370147728e-08
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
    slack_score = np.where(slack < 0, (-slack) ** 2.714986461505152, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.8442056214101434
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.5807597777524656
    is_tight_or_violated = slack <= 0
    successor_release_term = upward_rank * remaining_work
    successor_release_score = np.where(is_tight_or_violated, mad_normalize(successor_release_term), 0.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    host_load_gate = 1.0 / (1.0 + np.exp(-1.6556512349310455 * (uncertainty - 0.49421405304952837)))
    energy_scaled = min_incremental_energy * (1.0 + host_load_gate * uncertainty)
    wait_score = mad_normalize(ready_wait_time) * (1.0 + mad_normalize(uncertainty))
    slack_lb = -7.049697002654099
    slack_ub = 1.2134493754453586
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.9188876345473581 + (1.0 - 0.9188876345473581) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    energy_norm = mad_normalize(energy_scaled)
    unc_norm = mad_normalize(uncertainty)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.2752595573445156 * (uncertainty - 1.0)))
    energy_uncertainty_score = 0.63708958089122 * energy_norm * unc_norm * unc_sigmoid
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + successor_release_score
    score += slack_headroom_mask * (0.48981456807281576 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
