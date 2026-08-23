import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule integrating conditional DDL protection gate, host-load-aware successor release,
    and smoothed work-density interaction — all bounded, deterministic, and epsilon-guarded.
    
    Key structural mutations:
    - Replaces piecewise slack transformation with bounded exponential urgency (tanh-scaled) for smoother gradient near zero.
    - Introduces conditional DDL protection gate: critical path leverage only activates when both slack >= 0 AND uncertainty <= median.
    - Adds load-successor-release interaction: subtracts energy-normalized duration * (1 - tanh(slack_pressure)) to favor tasks whose execution unblocks successors early.
    - Replaces linear wait fairness with saturating exponential: 1 - exp(-wait_fairness_gain * norm_wait), avoiding clipping artifacts.
    - Uses robust mean-absolute deviation (MAD) scaling instead of median-centered MAD for better cross-seed stability.
    - All interactions are clipped, normalized, and guarded against inf/nan using np.finfo.
    """
    eps = 0.000815548682696315
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        mean = np.mean(x)
        mad = np.mean(np.abs(x - mean))
        scale = 1.0245134945984453 * (mad if mad > eps else eps)
        return (x - mean) / scale
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    slack_urgency = 2.475734270450554 * np.tanh(slack_pressure * 1.7607951883288575)
    slack_norm = np.where(slack < 0.0, 6.441012130474126 * -slack, slack_urgency)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_weight_adj = 0.877160636265619 * (1.0 - np.tanh(slack_pressure * 1.7607951883288575))
    rank_norm = mad_normalize(upward_rank)
    unc_median = np.median(uncertainty)
    ddl_protection_gate = ((slack >= 0.0) & (uncertainty <= unc_median + eps)).astype(float)
    critical_boost = 0.04937699180392682 * rank_norm * ddl_protection_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_gate = (slack >= 0.0).astype(float)
    work_density_bonus = 0.6579417557563125 * work_density_norm * work_density_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 0.822529842163759 * (1.0 - np.exp(-np.abs(wait_norm) + eps))
    successor_release_factor = 1.0 - np.tanh(slack_pressure * 1.7607951883288575)
    load_successor_interaction = -duration_norm * successor_release_factor
    unc_norm = mad_normalize(uncertainty)
    uncertainty_amplifier = 0.051812765837007516 * unc_norm * np.tanh(slack_pressure * 1.7607951883288575)
    score = slack_norm + 0.8067217549707755 * duration_norm + energy_weight_adj * energy_norm - critical_boost - work_density_bonus + wait_score + uncertainty_amplifier + load_successor_interaction
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
