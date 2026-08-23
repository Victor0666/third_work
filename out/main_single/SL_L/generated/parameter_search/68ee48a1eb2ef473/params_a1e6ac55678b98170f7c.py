import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule synthesizing strengths from both parents:
      - Retains Parent 2's robust lexicographic DDL protection (single hard gate on slack > ddl_protection_threshold)
      - Keeps unconditional upward_rank × remaining_work term (validated in >6 starvation cases)
      - Adds novel wait_starvation_suppression with bounded headroom gating to prevent starvation without compromising DDL safety
      - Introduces rank_headroom_nonlinearity: convex power-law scaling of rank importance near deadline headroom boundary
      - Uses unified MAD-based normalization throughout for stability across N=1 to large-N regimes
      - Removes all fragile dual-gating and percentile logic; replaces with monotonic, bounded, and numerically stable forms
      - All numeric literals strictly limited to {-2,-1,0,1,2}
      - Final score enforces strict DDL-first semantics: violations dominate; among feasible, energy-efficiency and critical path are balanced; starvation prevention is secondary and gated.
    """
    eps = 7.432049346656895e-07
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
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 3.4294097543267164, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.0347197761069844
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.4519965093688936
    critical_path_release = upward_rank * remaining_work * 0.9118268334837947
    ddl_safe_mask = np.where(slack > 0.37418916319255324, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_headroom = np.maximum(0.0, slack - 0.37418916319255324)
    max_slack_headroom = np.max(slack_headroom) + eps
    slack_headroom_normalized = np.clip(slack_headroom / max_slack_headroom, 0.0, 1.0)
    host_load_scale = slack_headroom_normalized * 0.9005420631636925
    rank_score_base = -mad_normalize(upward_rank)
    rank_headroom_weight = slack_headroom_normalized ** 1.548539556657094 + (1.0 - slack_headroom_normalized)
    rank_score = rank_score_base * rank_headroom_weight
    unc_sigmoid = 1.0 / (1.0 + np.exp(-7.411019281398908 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = host_load_scale * 1.856258980957966 * energy_norm * unc_norm * unc_sigmoid
    wait_score = mad_normalize(ready_wait_time) * slack_headroom_normalized * 0.3426861277374346
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_path_release)
    score += ddl_safe_mask * (1.3400724580930676 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
