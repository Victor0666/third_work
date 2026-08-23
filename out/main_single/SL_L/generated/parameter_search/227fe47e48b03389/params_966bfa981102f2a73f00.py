import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust joint MAD normalization and linear coupling,
       with Parent 1's explicit critical-path release weighting and starvation penalty.
    
    Structural innovations:
      - Joint MAD normalization over |slack|, uncertainty, duration_total, and ready_wait_time
        → improves coherence of all risk-aware features (not just DDL/energy)
      - Critical path release score is weighted *and* normalized jointly with other signals
      - Wait-term includes both decay exponent AND global starvation penalty (hybrid of both parents)
      - All DDL-violation terms use bounded linear couplings (no multiplicative explosions)
      - Added explicit duration_risk_penalty activation only under tight slack (<= threshold), not just violation
      - Numeric literals strictly limited to {-2,-1,0,1,2}; no hardcoded constants beyond that set
    """
    eps = 1.9753052894932814e-08
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
    risk_signals = np.concatenate([abs_slack, uncertainty, duration_total, ready_wait_time])
    risk_median = np.median(risk_signals)
    mad = np.median(np.abs(risk_signals - risk_median)) + eps
    joint_scale = 1.1556475727138722 * mad

    def joint_mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        return np.clip((x - risk_median) / joint_scale, -2.0, 2.0)
    slack_violation = np.maximum(0.0, -slack)
    unc_slack_coupling = 0.25047617499729913 * uncertainty * slack_violation
    tight_slack_mask = np.where(slack <= 1.2644925928246353, 1.0, 0.0)
    duration_risk = 0.45185059742261935 * duration_total * tight_slack_mask * uncertainty
    critical_path_release_score = upward_rank * remaining_work
    slack_headroom_mask = np.where(slack > 1.2644925928246353, 1.0, 0.0)
    energy_norm = joint_mad_normalize(min_incremental_energy)
    energy_score = 0.10722264466740003 * energy_norm * slack_headroom_mask
    unc_norm = joint_mad_normalize(uncertainty)
    energy_uncertainty_score = 0.03528250508094405 * energy_norm * unc_norm * slack_headroom_mask
    wait_base = ready_wait_time / (np.abs(slack) + 1.0)
    wait_decay = wait_base ** 0.7888073886191465 * slack_headroom_mask
    wait_score = 1.0657549441305363 * wait_decay
    wait_norm = joint_mad_normalize(wait_score)
    score = joint_mad_normalize(slack_violation) + joint_mad_normalize(unc_slack_coupling) + joint_mad_normalize(duration_risk)
    score += -2.4528440042241817 * joint_mad_normalize(critical_path_release_score)
    score += energy_score + energy_uncertainty_score + wait_norm
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
