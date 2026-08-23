import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's lexicographic safety with Parent 1's critical-path starvation mitigation:
      1. Retains Parent 2's tighter ddl_protection_gate_threshold for robust non-DDL activation.
      2. Reintroduces unconditional upward_rank × remaining_work interaction (evidence-backed bottleneck identification), but now masked by slack_headroom_mask for DDL safety.
      3. Unifies all normalization via MAD (robust for N=1 and sparse sets) and clips to [-2,2] using allowed literals.
      4. Anti-starvation uses wait-time scaled by total slack headroom — no new parameter introduced; reuses existing ddl_protection_gate_threshold logic.
      5. All divisions guarded; no inf/nan; deterministic; shape-(N,) guaranteed.
    """
    eps = 9.630834812021708e-08
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
    slack_score = np.where(slack < 0, (-slack) ** 2.0142062642316123, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.9560690693610502
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.6118027684974869
    slack_headroom_mask = np.where(slack > 0.1456423383769378, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    rank_work_interaction = upward_rank * remaining_work
    rank_work_score = -mad_normalize(rank_work_interaction)
    slack_lb = -22.306699764538422
    slack_ub = 25.505236479732027
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.16017433522814672 + (1.0 - 0.16017433522814672) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.3534680727641524 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.03859788949613988 * energy_norm * unc_norm * unc_sigmoid
    total_slack_headroom = np.sum(np.maximum(0.0, slack)) + eps
    wait_normalized = np.where(slack_headroom_mask > 0.0, ready_wait_time / (total_slack_headroom + 1.0), 0.0)
    wait_score = mad_normalize(wait_normalized)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk)
    score += slack_headroom_mask * (0.7809053449382385 * energy_eff_score + rank_work_score + rank_score + energy_uncertainty_score + wait_score)
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 2.4131077888832517, 1.0)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
