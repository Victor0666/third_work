import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with smooth slack penalty, tanh-based uncertainty gating, and MAD-normalization.
    
    Key structural improvements:
    - Replaces piecewise slack scoring with continuous `max(0,-slack)**p` — eliminates discontinuities and instability.
    - Replaces IQR normalization with robust MAD (mean absolute deviation) to avoid outlier sensitivity of median/IQR.
    - Replaces sigmoid uncertainty gates with bounded `tanh(steepness * uncertainty)` for smoother, monotonic modulation.
    - Applies uncertainty gating *only* to energy/duration terms (not slack_score), decoupling deadline urgency from risk scaling.
    - All inputs sanitized with `np.nan_to_num`; no in-place mutation; strict shape-(N,) output.
    """
    eps = 1.0063660326864922e-07
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
        center = np.mean(x)
        abs_dev = np.abs(x - center)
        scale = np.mean(abs_dev) + eps
        return (x - center) / (scale + eps)
    slack_penalty = np.maximum(0.0, -slack) ** 1.4475687961540822
    duration_total = min_exec_time + min_comm_time + eps
    duration_norm = mad_normalize(duration_total)
    unc_gate = np.tanh(4.845187311731391 * uncertainty)
    duration_risk = duration_norm * (1.0 + unc_gate * 1.4724316509404018)
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_norm = mad_normalize(energy_per_sec)
    energy_eff_score = energy_eff_norm * 0.9855726635417797 * (1.0 + unc_gate)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = mad_normalize(uncertainty * deadline_pressure) * 0.2730394716932367
    rank_norm = mad_normalize(upward_rank)
    clipped_slack = np.maximum(0.0, -slack)
    critical_bonus = rank_norm * (1.0 + 0.7583874630975052 * clipped_slack)
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.7673682006956977 * energy_norm * unc_norm * unc_gate
    energy_base_score = mad_normalize(min_incremental_energy) * (1.0 + unc_gate)
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 0.6328935301153771 * (1.0 + np.tanh(0.2691336545081525 * wait_norm))
    duration_q = np.quantile(duration_total, 0.9424142677904965) + eps
    work_density = remaining_work / duration_q
    work_density_norm = mad_normalize(work_density)
    work_density_bonus = 0.9812495439014353 * work_density_norm
    score = mad_normalize(slack_penalty) + duration_risk + energy_eff_score + unc_slack_coupling + energy_uncertainty_score + energy_base_score + wait_boost - critical_bonus - work_density_bonus
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
