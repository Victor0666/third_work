import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's smooth gating and exponential wait boost with Parent 1's energy-efficiency ratio.
    Smaller score = higher priority. All numeric literals are -2,-1,0,1,2 or derived from PARAMS."""
    eps = 0.09536502092712311
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_norm(x):
        x_abs = np.abs(x)
        denom = np.mean(x_abs) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    norm_duration = robust_norm(min_exec_time + min_comm_time)
    norm_rank = robust_norm(upward_rank)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    slack_power = np.sign(norm_slack) * np.abs(norm_slack) ** 2.7049053991551646
    slack_normalized_for_gate = (norm_slack - 0.17044666928439578) / (1.0 + eps)
    gate_activation = np.clip(-slack_normalized_for_gate, 0.0, 1.0)
    dur_uncert_coupling = np.clip(norm_duration * norm_uncert, 0.0, 1.0) * 0.8365486224738944
    wait_boost = 1.0 - np.exp(-0.003942236162035342 * (norm_wait + eps))
    slack_pressure_mask = np.where(norm_slack < 0, 1.0, 0.0)
    uncert_slack_penalty = norm_uncert * slack_pressure_mask * 1.1893886831135934
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = robust_norm(energy_per_sec)
    score = slack_power
    safe_slack_mask = np.where(norm_slack > 0, 1.0, 0.0)
    score += safe_slack_mask * 2.006319910059489 * norm_energy
    score += gate_activation * 0.7334341380490116 * -norm_rank
    score += dur_uncert_coupling
    score -= wait_boost * 0.23236083681617897
    score += uncert_slack_penalty
    score += safe_slack_mask * energy_eff_score
    score = np.nan_to_num(score, nan=-246142.71238154313, posinf=106431.88676321114, neginf=-98059547.51120949)
    return score
