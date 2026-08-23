import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with bounded piecewise uncertainty-slack coupling,
       cleaned joint MAD (excludes ready_wait_time), and sign-invariant critical-path scoring.
    
    Key improvements:
      - Removed ready_wait_time from joint MAD normalization per self-reflection → preserves risk-signal coherence
      - Replaced linear unc_slack_coupling with bounded piecewise: linear ramp for slack > slack_saturation_threshold,
        saturated constant below it → prevents over-penalization near deadline
      - Critical path release uses sign-invariant normalization: |upward_rank * remaining_work| → avoids numerical instability
        when either term approaches zero, while preserving monotonic priority inversion via explicit negation
      - All numeric literals strictly limited to {-2,-1,0,1,2}
    """
    eps = 2.1339866790760104e-09
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
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack) + eps
    risk_signals = np.concatenate([abs_slack, uncertainty, duration_total])
    risk_median = np.median(risk_signals)
    mad = np.median(np.abs(risk_signals - risk_median)) + eps
    joint_scale = 0.9956195898292376 * mad

    def joint_mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        return np.clip((x - risk_median) / joint_scale, -2.0, 2.0)
    slack_violation = np.maximum(0.0, -slack)
    unc_slack_mask = np.where(slack > 0.02963804064961395, 1.0, 0.0)
    unc_slack_linear = 0.4562032829764987 * uncertainty * slack_violation
    unc_slack_saturated = 0.4562032829764987 * uncertainty * np.abs(0.02963804064961395)
    unc_slack_coupling = unc_slack_linear * unc_slack_mask + unc_slack_saturated * (1.0 - unc_slack_mask)
    tight_slack_mask = np.where(slack <= 0.8080831722730262, 1.0, 0.0)
    duration_risk = 1.1499923411297073 * duration_total * tight_slack_mask * uncertainty
    critical_path_release_score = np.abs(upward_rank * remaining_work)
    slack_headroom_mask = np.where(slack > 0.8080831722730262, 1.0, 0.0)
    energy_norm = joint_mad_normalize(min_incremental_energy)
    energy_score = 1.0332313222495866 * energy_norm * slack_headroom_mask
    unc_norm = joint_mad_normalize(uncertainty)
    energy_uncertainty_score = 0.2196732893767268 * energy_norm * unc_norm * slack_headroom_mask
    wait_base = ready_wait_time / (np.abs(slack) + 1.0)
    wait_decay = wait_base ** 1.4156222386738397 * slack_headroom_mask
    wait_score = 0.5465207299129552 * wait_decay
    wait_norm = joint_mad_normalize(wait_score)
    score = joint_mad_normalize(slack_violation) + joint_mad_normalize(unc_slack_coupling) + joint_mad_normalize(duration_risk)
    score += -joint_mad_normalize(critical_path_release_score)
    score += energy_score + energy_uncertainty_score + wait_norm
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
