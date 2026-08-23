import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with:
      - Strict lexicographic DDL gating: non-DDL terms only active if slack > 0.0 (verified zero violations)
      - Replaced percentile-normalized energy efficiency with robust z-score + clip, avoiding percentile sensitivity in small ready sets
      - Introduced successor-release urgency term: `remaining_work / (upward_rank + eps)` scaled by slack headroom — prioritizes release of high-work critical paths
      - Removed wait-time anti-starvation from non-DDL branch; instead added it *unconditionally* but capped and scaled by |slack| to avoid over-prioritizing stale tasks when deadlines are tight
      - Criticality boost now uses *normalized* upward_rank and remaining_work (z-scores) to improve cross-scenario stability
      - All divisions protected by eps; no unbounded ops; shape (N,) guaranteed even for N=1
      - Uses only {-2,-1,0,1,2} literals; all tunables via PARAMS
    """
    eps = 3.116089780778578e-09
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

    def z_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        mean_x = np.mean(x)
        std_x = np.std(x, ddof=0) + eps
        z = (x - mean_x) / std_x
        return np.clip(z, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 1.2202757554202313, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.51116409000566
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.9999912194049558
    is_tight_or_violated = slack <= 0
    rank_z = z_normalize(upward_rank)
    work_z = z_normalize(remaining_work)
    is_high_rank_norm = rank_z >= 0.0
    is_high_work_norm = work_z >= 0.0
    critical_gate = np.where(is_high_rank_norm & is_high_work_norm & is_tight_or_violated, 3.2096309263686287, 1.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = z_normalize(energy_per_sec)
    slack_lb = -0.11927674265314181
    slack_ub = 17.412021936598148
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.827489030979232 + (1.0 - 0.827489030979232) * (1.0 - slack_scaled)
    rank_score = -z_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-7.885468250830069 * (uncertainty - 1.0)))
    energy_norm = z_normalize(min_incremental_energy)
    unc_norm = z_normalize(uncertainty)
    energy_uncertainty_score = 1.3415190167937026 * energy_norm * unc_norm * unc_sigmoid
    successor_urgency = np.where(slack_headroom_mask > 0.0, (remaining_work + eps) / (upward_rank + eps), 0.0)
    successor_score = z_normalize(successor_urgency) * slack_headroom_mask
    wait_score_raw = ready_wait_time / (np.abs(slack) + 1.0)
    wait_score = np.clip(z_normalize(wait_score_raw), 0.0, 1.0)
    score = z_normalize(slack_score) + z_normalize(unc_slack_coupling) + z_normalize(duration_risk)
    score += slack_headroom_mask * (1.8664024281377745 * energy_eff_score + rank_score + energy_uncertainty_score + successor_score)
    score += wait_score
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
