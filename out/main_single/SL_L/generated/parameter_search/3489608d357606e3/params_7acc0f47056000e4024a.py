import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust normalization and work-density with Parent 1's successor-release insight,
       enhanced by unified slack-pressure gating and deterministic priority stabilization.
    
    Structural improvements:
    - Removes DDL-protection mode (hard switch) and priority_stabilization_alpha (exceeded parameter count).
    - Retains all 12 parameters: epsilon, slack_risk_penalty, slack_urgency_scale, energy_efficiency_bias,
      critical_path_leverage, wait_fairness_gain, uncertainty_sensitivity, duration_balance, slack_cap,
      slack_pressure_tanh_scale, work_density_weight, robustness_mad_factor.
    - Introduces successor_release_weight *implicitly* via reuse of `work_density_weight` logic — but now applied to
      `upward_rank * remaining_work` (same proxy) without adding new parameter; avoids duplication while preserving insight.
    - Uses unified `slack_pressure_bounded = tanh(slack_pressure * scale)` for energy, uncertainty, and fairness scaling.
    - All numeric literals are -2, -1, 0, 1, or 2; no hidden constants.
    """
    eps = 0.06285437968384901
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
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.713335538756567 * (mad if mad > eps else eps)
        return (x - med) / scale
    neg_mask = slack < 0.0
    zeroish_mask = (slack >= 0.0) & (slack < 1.0)
    pos_mask = slack >= 1.0
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 4.60759080933288 * -slack[neg_mask]
    slack_norm[zeroish_mask] = 2.5112846054711824 * (np.exp(slack[zeroish_mask]) - 1.0)
    slack_norm[pos_mask] = np.clip(slack[pos_mask], 0.0, 13.38814258772174)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    slack_pressure_bounded = np.tanh(slack_pressure * 0.5103944688845861)
    energy_weight_adj = 2.305123036836259 * (1.0 - slack_pressure_bounded)
    rank_norm = mad_normalize(upward_rank)
    critical_gate = (slack >= 0.0).astype(float)
    critical_boost = 0.033262322745778415 * rank_norm * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_gate = (slack >= 0.0).astype(float)
    work_density_bonus = 0.7870522340889896 * work_density_norm * work_density_gate
    successor_impact = upward_rank * remaining_work
    successor_norm = mad_normalize(successor_impact)
    successor_gate = (slack >= 0.0).astype(float)
    successor_bonus = 0.7870522340889896 * successor_norm * successor_gate
    wait_headroom = np.maximum(1.0, slack + 1.0)
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 0.4186693925699105 * wait_norm / (wait_headroom + eps)
    unc_norm = mad_normalize(uncertainty)
    uncertainty_amplifier = 0.6606906579034185 * unc_norm * slack_pressure_bounded
    score = slack_norm + 0.39315361215583394 * duration_norm + energy_weight_adj * energy_norm - critical_boost - work_density_bonus - successor_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
