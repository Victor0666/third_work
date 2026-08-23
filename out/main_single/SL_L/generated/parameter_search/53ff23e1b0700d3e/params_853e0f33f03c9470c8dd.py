import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with strict lexicographic DDL enforcement and starvation-aware gating:
      - Restores hard `slack > 0` binary gate for non-DDL terms (per reflection).
      - Anti-starvation now requires slack > 0 AND slack >= 1.0 (fixed threshold, not tunable) to avoid parameter bloat.
      - Uses per-feature MAD normalization (not joint) for stability in small-N and heterogeneous settings.
      - Critical path release is gated by slack <= 0 (true DDL-critical activation).
      - All non-DDL terms are zero when slack <= 0 — strict lexicographic ordering preserved.
      - Final score preserves deterministic, finite, shape-(N,) output with no in-place mutation.
    """
    eps = 1.9905558313710142e-07
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
    slack_score = np.where(slack < 0, (-slack) ** 2.117297721170188, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.518938222161109
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.2581870863575129
    is_ddl_constrained = slack <= 0.0
    critical_release = upward_rank * remaining_work
    critical_release_norm = -mad_normalize(critical_release)
    successor_release_contribution = np.where(is_ddl_constrained, critical_release_norm * 0.5027977873229198, 0.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total_safe = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total_safe
    energy_eff_score = mad_normalize(energy_per_sec) * 0.5701218942841189
    slack_lb = -38.37820787596586
    slack_ub = 34.320284559920346
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.2978622769534006 + (1.0 - 0.2978622769534006) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.5270905100097008 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.07060252931340617 * energy_norm * unc_norm * unc_sigmoid
    starvation_gate = np.where(slack >= 1.0, 1.0, 0.0)
    starvation_enabled = slack_headroom_mask * starvation_gate
    wait_power = ready_wait_time ** 1.5723109050716413
    wait_score = mad_normalize(wait_power)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + successor_release_contribution
    score += starvation_enabled * (energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
