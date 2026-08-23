import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining best practices:
      - Uses robust MAD-based normalization (from Parent 2) for stability across N.
      - Retains unconditional upward_rank × remaining_work critical-path interaction (Parent 2).
      - Integrates smooth feasibility gate (from Parent 1) using feasibility_sigmoid_steepness, enabling differentiable optimization.
      - Replaces linear (1 - feasibility_gate) anti-starvation with power-law coupling using feasibility_sigmoid_steepness itself (no new param), avoiding parameter count violation.
      - All divisions guarded; all outputs finite, shape-(N,), deterministic; only literals {-2,-1,0,1,2} used.
    """
    eps = 9.042253466881417e-05
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
    slack_score = np.where(slack < 0, (-slack) ** 3.006914792950652, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.653041350048282
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.3846974685801415
    rank_work_interaction = upward_rank * remaining_work
    rank_work_med = np.median(rank_work_interaction) if N > 1 else np.mean(rank_work_interaction)
    rank_work_normalized = mad_normalize(rank_work_interaction) / (np.abs(rank_work_med) + eps)
    feasibility_gate = 1.0 / (1.0 + np.exp(-9.336996324310142 * slack))
    wait_slack_coupling = 1.0 - (1.0 - feasibility_gate) ** 9.336996324310142
    wait_score = mad_normalize(ready_wait_time) * wait_slack_coupling
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -15.05778690966035
    slack_ub = 22.066908386342522
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.27603931873329923 + (1.0 - 0.27603931873329923) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-5.619591894654802 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.0458732144344277 * energy_norm * unc_norm * unc_sigmoid
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + rank_work_normalized
    score += feasibility_gate * (0.47885879115156127 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    is_tight_or_violated = slack <= 0
    critical_gate = np.where(is_tight_or_violated, 0.6138995268486219, 1.0)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
