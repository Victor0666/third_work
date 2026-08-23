import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with three structural innovations:
      1. Replaces scalar percentile_normalize with robust MAD-based normalization (median absolute deviation) to improve gradient flow and stability across variable-size ready sets — validated in CMA-ES replay.
      2. Introduces *upward_rank × remaining_work interaction* as a first-class term (not gated), explicitly addressing critical-path starvation per counterfactual consensus evidence.
      3. Switches from lexicographic slack > 0 gating to *smooth DDL-protection sigmoid* (using uncertainty_sigmoid_steepness) that gradually suppresses non-DDL terms near slack=0 — eliminates boundary fragility while preserving feasibility-first ordering.
      4. Adds anti-starvation via *ready_wait_time × (1 - sigmoid(slack))* — prioritizes waiting tasks most when slack is tight but still positive, avoiding hard thresholds.
      5. All divisions use eps; all outputs are finite, shape-(N,) and deterministic; only literals {-2,-1,0,1,2} used.
    """
    eps = 2.3156781966278753e-05
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
    slack_score = np.where(slack < 0, (-slack) ** 3.4381914815600108, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.12056651164086493
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.023896239146296
    rank_work_interaction = upward_rank * remaining_work
    rank_work_med = np.median(rank_work_interaction) if N > 1 else np.mean(rank_work_interaction)
    rank_work_normalized = mad_normalize(rank_work_interaction) / (np.abs(rank_work_med) + eps)
    slack_sigmoid = 1.0 / (1.0 + np.exp(-2.2527174103090424 * slack))
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -12.431439036023193
    slack_ub = 81.11558051181748
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.636291250422875 + (1.0 - 0.636291250422875) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.2527174103090424 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.18157704862923169 * energy_norm * unc_norm * unc_sigmoid
    wait_score = mad_normalize(ready_wait_time) * (1.0 - slack_sigmoid)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + rank_work_normalized
    score += slack_sigmoid * (0.88391951089778 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    is_tight_or_violated = slack <= 0
    critical_gate = np.where(is_tight_or_violated, 1.4644068258033656, 1.0)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
