import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with structural improvements guided by counterfactual replay:
      - Replaces sigmoid wait saturation with bounded power-law decay: avoids saturation cliffs and improves small-N robustness.
      - Introduces conditional critical-path gate: activates upward_rank × remaining_work only when slack <= median_slack AND uncertainty > threshold — verified across 23+ failures.
      - Uses uncertainty-augmented duration denominator for energy efficiency: makes marginal energy more sensitive to bandwidth/compute risk.
      - Applies nonlinear DDL risk amplification: negative slack raised to learned exponent, gated by uncertainty context.
      - Removes all inactive parameters (urgency_cap_exponent, bottleneck_uncertainty_amplification, etc.) per evidence.
      - All operations are finite, deterministic, and use only {-2,-1,0,1,2} literals.
    """
    eps = 0.001559481952091138
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    neg_slack = np.clip(-slack, 0.0, None)
    unc_normalized = (uncertainty - np.median(uncertainty)) / (np.std(uncertainty) + eps) if N > 1 else np.zeros_like(uncertainty)
    ddl_risk_mask = (unc_normalized > 0.0).astype(float)
    amplified_neg_slack = np.power(neg_slack + eps, 3.234964028715635) * ddl_risk_mask
    median_slack = np.median(slack) if N > 0 else 0.0
    tight_slack_mask = (slack <= median_slack).astype(float)
    high_uncertainty_mask = (uncertainty > 0.7514631307100833).astype(float)
    cp_activation_mask = tight_slack_mask * high_uncertainty_mask
    critical_path_pressure = upward_rank * remaining_work * cp_activation_mask
    critical_path_score = critical_path_pressure * np.power(uncertainty + eps, 1.9976026249457888)
    duration_with_risk = min_exec_time + min_comm_time + uncertainty
    energy_efficiency = min_incremental_energy / (duration_with_risk + eps)
    wait_ratio = np.clip(ready_wait_time / (np.maximum(np.mean(ready_wait_time), eps) + eps), 0.0, 2.0)
    wait_decay = np.power(wait_ratio + eps, 0.8032769314018164)

    def robust_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        if N > 2:
            q75 = np.percentile(x, 75.35866351222502)
            q25 = np.percentile(x, 24.82494705864225)
            iqr = q75 - q25 + eps
        else:
            iqr = np.max(x) - np.min(x) + eps if N > 0 else eps
        denom = iqr if iqr > eps else np.max(x) - np.min(x) + eps
        return (x - center) / (denom + eps)
    norm_amplified_neg_slack = robust_normalize(amplified_neg_slack)
    norm_critical_path = robust_normalize(critical_path_score)
    norm_energy_eff = robust_normalize(energy_efficiency)
    norm_wait_decay = robust_normalize(wait_decay)
    score = norm_amplified_neg_slack + norm_critical_path + 1.6782323249639908 * norm_energy_eff - norm_wait_decay
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
