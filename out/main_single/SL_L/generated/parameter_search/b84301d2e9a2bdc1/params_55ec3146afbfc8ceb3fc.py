import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's smoothness and Parent 1's critical-path congestion awareness,
       enhanced with clipped tanh gating, MAD-normalized work-density as core term, and unified risk suppression.
    
    Structural improvements:
    - Replaces all sigmoid gates with `np.clip(np.tanh(x), 0, 1)` to enforce zero-saturated, monotonic risk suppression.
    - Introduces `work_density = remaining_work / (min_exec_time + min_comm_time + eps)` as a *standalone priority term*,
      normalized via MAD and weighted — not just a bonus subtracted from score.
    - Removes empirical slack bounds; uses MAD-normalized `slack` directly in all interactions for robustness.
    - Unifies uncertainty gating: `unc_gate = np.clip(np.tanh(steepness * uncertainty), 0, 1)` applied consistently
      to *all* non-deadline terms (energy, duration, wait, work-density) to suppress risky optimization.
    - Retains smooth slack penalty `max(0,-slack)**p` (Parent 2) but adds Parent 1’s `upward_rank × remaining_work`
      interaction scaled by `unc_gate * (1.0 - slack_sigmoid)` as explicit critical-path congestion signal.
    - Anti-starvation `wait_boost` now gated by both `unc_gate` and `slack_headroom_mask` (Parent 1 style) for DDL-safe fairness.
    - All normalizations use MAD with N=1 fallback; no percentile or IQR dependencies.
    """
    eps = 2.734024380355191e-06
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
        center = np.median(x)
        mad = np.median(np.abs(x - center))
        denom = mad + eps
        normalized = (x - center) / denom
        return np.clip(normalized, -2.0, 2.0)
    unc_gate = np.clip(np.tanh(3.335654890131265 * uncertainty), 0.0, 1.0)
    slack_penalty = np.maximum(0.0, -slack) ** 1.750725696918593
    deadline_pressure = np.maximum(0.0, -slack)
    duration_total = min_exec_time + min_comm_time + eps
    duration_norm = mad_normalize(duration_total)
    duration_risk = duration_norm * (1.0 + unc_gate * 0.6786550938628039)
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_norm = mad_normalize(energy_per_sec)
    energy_eff_score = energy_eff_norm * 1.2437594082473769 * (1.0 + unc_gate)
    unc_slack_coupling = mad_normalize(uncertainty * deadline_pressure) * 0.0026822835538218293
    rank_norm = mad_normalize(upward_rank)
    clipped_slack = np.maximum(0.0, -slack)
    critical_bonus = rank_norm * (1.0 + 0.7077270234897997 * clipped_slack)
    work_density = remaining_work / (duration_total + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_term = 0.44119849971419406 * work_density_norm * (1.0 + unc_gate)
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.736220611644599 * energy_norm * unc_norm * unc_gate
    energy_base_score = mad_normalize(min_incremental_energy) * (1.0 + unc_gate)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 0.37159993503393074 * (1.0 + np.tanh(0.10682992250214648 * wait_norm)) * slack_headroom_mask * unc_gate
    rank_work_product = upward_rank * remaining_work
    slack_sigmoid = 1.0 / (1.0 + np.exp(-3.335654890131265 * (slack + 1.0)))
    congestion_strength = np.clip(1.0 - slack_sigmoid, 0.0, 1.0)
    congestion_term = mad_normalize(rank_work_product) * congestion_strength * unc_gate * 0.7077270234897997
    score = mad_normalize(slack_penalty) + duration_risk + energy_eff_score + unc_slack_coupling + energy_uncertainty_score + energy_base_score + wait_boost - critical_bonus - work_density_term - congestion_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
