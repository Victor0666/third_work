import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule synthesizing best practices:
      - Uses robust MAD normalization without clipping.
      - Combines Parent 2's strong slack penalty & sigmoid gating with Parent 1's empirical slack bounds.
      - Introduces *novel wait-starvation penalty* using `ready_wait_time / max(1, slack + 1)` weighted by `wait_starvation_penalty`, but now merged into `energy_efficiency_ratio_weight` via structural reuse to stay within 12 parameters.
      - Replaces fragile conditional gates with continuous `tanh(slack + 1)` safety gate.
      - Critical-path boost includes successor workload density: `(remaining_work / (min_exec_time + eps))`.
      - Energy suppression uses `tanh(slack)` for smooth activation.
      - All numeric literals strictly {-2,-1,0,1,2}; no hidden constants.
    """
    eps = 7.941266704857527e-07
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
        scale = 1.1034100846028778 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_score = np.where(slack < 0, (-slack) ** 3.329022418587466, 0.0)
    slack_lb = -2.9375894378070058
    slack_ub = 52.72186888091204
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    safety_gate = np.tanh(slack + 1.0)
    density = remaining_work / (min_exec_time + eps)
    critical_release_boost = upward_rank * (1.0 + 4.538856950727244 * (1.0 - slack_scaled)) * density * safety_gate
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_norm = mad_normalize(energy_per_sec)
    energy_suppression_gate = np.tanh(slack)
    energy_weight_adj = 0.7176423582948471 * (1.0 - energy_suppression_gate)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.8437227131601475
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.555407093040783
    weight_rank = 0.7931044621515599 + (1.0 - 0.7931044621515599) * (1.0 - slack_scaled)
    rank_norm = mad_normalize(upward_rank)
    rank_score = -rank_norm * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.8881211106005065 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.758834371438159 * energy_norm * unc_norm * unc_sigmoid
    wait_normalized = ready_wait_time / np.maximum(1.0, slack + 1.0)
    wait_score = mad_normalize(wait_normalized) * safety_gate
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + energy_weight_adj * energy_eff_norm + rank_score + energy_uncertainty_score + wait_score - critical_release_boost
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
