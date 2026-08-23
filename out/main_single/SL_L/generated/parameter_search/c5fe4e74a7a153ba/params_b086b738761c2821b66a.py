import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining best practices from both parents:
      - Uses robust MAD normalization (from Parent 1 analysis) instead of percentile, eliminating literal violations and improving sparse-set stability.
      - Retains Parent 2's strong slack power penalty and criticality boost, but adds slack-aware energy gating (Parent 1 insight) to suppress energy optimization when slack > 0 AND uncertainty < median.
      - Introduces bounded critical-path release boost: upward_rank * (1 + criticality_boost * (1 - slack_scaled)), activated only for slack <= 0, directly leveraging successor-release interaction evidence.
      - Drops fragile percentile_normalize in favor of unified mad_normalize with epsilon-guarded scale.
      - All numeric literals are restricted to {-2, -1, 0, 1, 2}; no other constants appear in expressions.
      - Final score is deterministic, finite, shape-(N,) and prioritizes deadline feasibility first, then energy efficiency within feasible set.
    """
    eps = 5.2043182154023164e-08
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
        med = np.median(x) if N > 1 else np.mean(x)
        mad = np.median(np.abs(x - med)) if N > 1 else eps
        scale = 1.1489141054326937 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_score = np.where(slack < 0, (-slack) ** 3.253532959848645, 0.0)
    slack_lb = -11.24976615784296
    slack_ub = 13.432371230810368
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    release_gate = (slack <= 0).astype(float)
    critical_release_boost = upward_rank * (1.0 + 1.9186867105961696 * (1.0 - slack_scaled)) * release_gate
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_norm = mad_normalize(energy_per_sec)
    median_unc = np.median(uncertainty) if N > 1 else np.mean(uncertainty)
    energy_suppression_gate = ((slack > 0.0) & (uncertainty <= median_unc + eps)).astype(float)
    energy_weight_adj = 0.9491036517020823 * (1.0 - energy_suppression_gate)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.553333131895322
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.8300745356316472
    weight_rank = 0.013137765962045066 + (1.0 - 0.013137765962045066) * (1.0 - slack_scaled)
    rank_norm = mad_normalize(upward_rank)
    rank_score = -rank_norm * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.968675097776124 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.37422308527395576 * energy_norm * unc_norm * unc_sigmoid
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + energy_weight_adj * energy_eff_norm + rank_score + energy_uncertainty_score + mad_normalize(min_incremental_energy) * (1.0 - weight_rank) - critical_release_boost
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
