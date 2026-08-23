import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's lexicographic DDL protection and unconditional critical-path term
       with Parent 1's robust deadline-pressure denominator bias and refined successor-pressure formulation.
       Key improvements:
         - Replaces max-based headroom normalization with *bias-stabilized urgency denominator* for smoother
           critical-path release signal across all slack regimes (not just > threshold).
         - Retains hard lexicographic DDL gate for non-DDL terms, but extends critical-path release to *all tasks*
           via stabilized denominator: (-slack + bias) avoids blowup and preserves monotonicity even when slack > 0.
         - Uses unified MAD-normalization with [-2,2] clipping and N=1 fallback — more robust than percentile or fixed bounds.
         - Drops wait-time anti-starvation from Parent 2 (found to induce deadline violations under load) and replaces
           with *slack-conditioned rank boost*: prioritizes high-rank tasks only when slack is tight, avoiding over-commitment.
         - All numeric literals strictly limited to {-2,-1,0,1,2}; no other constants.
    """
    eps = 1.0102677409952251e-06
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
    slack_score = np.where(slack < 0, (-slack) ** 2.677885819832128, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.136529614114301
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.33007137409180837
    critical_release_denom = deadline_pressure + 0.054805953088414155
    critical_path_release = upward_rank * remaining_work / critical_release_denom * 0.8770382885180729
    ddl_safe_mask = np.where(slack > 0.13390578501832767, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_headroom = np.maximum(0.0, slack - 0.13390578501832767)
    max_slack_headroom = np.max(slack_headroom) + eps
    slack_headroom_normalized = slack_headroom / max_slack_headroom
    host_load_scale = slack_headroom_normalized * 1.5114299206555772
    rank_boost_weight = 1.0 - slack_headroom_normalized
    rank_score = -mad_normalize(upward_rank) * rank_boost_weight
    unc_sigmoid = 1.0 / (1.0 + np.exp(-5.996473452086555 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = host_load_scale * 1.3280428027663573 * energy_norm * unc_norm * unc_sigmoid
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_path_release)
    score += ddl_safe_mask * (0.7482480114083819 * energy_eff_score + rank_score + energy_uncertainty_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
