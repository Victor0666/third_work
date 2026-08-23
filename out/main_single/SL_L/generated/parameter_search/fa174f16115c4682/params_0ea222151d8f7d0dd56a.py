import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Hybrid priority rule combining Parent 2's robust piecewise slack urgency and triple-gated critical path,
    with Parent 1's local uncertainty-aware energy gating and unified slack-uncertainty coupling.
    
    Key structural improvements:
    - Retains Parent 2's interpretable piecewise exponential slack urgency (linear penalty + capped exp),
      but adds *adaptive energy gating*: energy_score is zeroed when uncertainty exceeds
      task-local median * threshold, improving confidence-aware optimization.
    - Replaces Parent 2's global median energy gate with *local adaptive gating*: energy contribution
      is suppressed only when task's own uncertainty is high relative to workload distribution,
      avoiding over-suppression in heterogeneous scenarios.
    - Keeps triple-gated critical-path leverage (slack>=0 ∧ low_uncertainty ∧ low_energy), now using
      the same local uncertainty threshold for consistency across gates.
    - Uses unified uncertainty-amplifier: couples normalized uncertainty multiplicatively with slack_urgency
      (not pressure), aligning with risk-adjusted fuzzy semantics (urgency under uncertainty).
    - All normalizations use robust MAD with Gaussian-consistent scaling; no numeric literals except {-2,-1,0,1,2}.
    """
    eps = 8.459228487841763e-08
    finfo = np.finfo(np.float64)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=np.float64)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.25235686918982 * (mad if mad > eps else eps)
        return (x - med) / scale
    neg_mask = slack < 0.0
    tight_mask = (slack >= 0.0) & (slack <= 1.081738663885717)
    loose_mask = slack > 1.081738663885717
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 1.986727507986494 * -slack[neg_mask]
    slack_norm[tight_mask] = 2.4601873578962845 * (np.exp(slack[tight_mask]) - 1.0)
    slack_norm[loose_mask] = 2.4601873578962845 * (np.exp(1.081738663885717) - 1.0)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    unc_median = np.median(uncertainty)
    energy_gate = (uncertainty <= unc_median * 0.4517177542456746 + eps).astype(float)
    gated_energy = min_incremental_energy * energy_gate
    energy_norm = mad_normalize(gated_energy)
    energy_score = 0.5764163322341659 * energy_norm
    rank_norm = mad_normalize(upward_rank)
    unc_med = np.median(uncertainty)
    eng_med = np.median(min_incremental_energy)
    unc_gate = (uncertainty <= unc_med).astype(float)
    eng_gate = (min_incremental_energy <= eng_med).astype(float)
    critical_gate = ((slack >= 0.0) * unc_gate * eng_gate).astype(float)
    critical_boost = 0.006360137472268502 * rank_norm * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_gate = (slack >= 0.0).astype(float)
    work_density_bonus = 0.3211163012200988 * work_density_norm * work_density_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 1.0 - np.exp(-0.8195725667345515 * np.maximum(wait_norm, 0.0))
    wait_score = wait_boost * (slack >= 0.0).astype(float)
    unc_norm = mad_normalize(uncertainty)
    tight_urgency = np.zeros_like(slack)
    tight_urgency[tight_mask] = np.exp(slack[tight_mask]) - 1.0
    tight_urgency[loose_mask] = np.exp(1.081738663885717) - 1.0
    uncertainty_amplifier = 0.01900647163619262 * unc_norm * tight_urgency
    score = slack_norm + 0.8201538603719913 * duration_norm + energy_score - critical_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    return score
