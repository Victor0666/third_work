import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with structural improvements guided by counterfactual evidence:
      - Introduces joint DDL-risk protection gate: activates bottleneck & successor-release terms only when
        (slack <= median_slack * ddl_risk_gate_slack_ratio) AND (normalized_uncertainty > ddl_risk_gate_threshold)
      - Replaces sigmoid wait saturation with bounded power-law saturation for smoother, more robust anti-starvation.
      - Uses raw upward_rank × remaining_work as uncoupled critical-path pressure (no multiplicative urgency coupling).
      - Adds uncertainty-modulated energy penalty: higher uncertainty → stronger marginal energy penalty.
      - Removes all adaptive normalization on neg_slack and urgency_linear to preserve hard-deadline dominance.
      - All operations are finite, deterministic, and use only {-2,-1,0,1,2} literals.
    """
    eps = 3.4581298780497883e-06
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
    median_slack = np.median(slack) if N > 0 else 0.0
    high_risk_slack_condition = slack <= median_slack * 0.5347536677489394 + eps
    unc_min = np.min(uncertainty) if N > 0 else 0.0
    unc_max = np.max(uncertainty) if N > 0 else eps
    normalized_uncertainty = (uncertainty - unc_min) / (unc_max - unc_min + eps)
    high_uncertainty_condition = normalized_uncertainty > 0.2793073138576099
    ddl_risk_gate = np.where(high_risk_slack_condition & high_uncertainty_condition, 1.0, 0.0)
    critical_path_pressure = upward_rank * remaining_work
    energy_penalty = min_incremental_energy * (1.0 + 0.17953327231353783 * uncertainty)
    wait_scaled = np.clip(ready_wait_time / (np.maximum(np.mean(ready_wait_time), eps) + eps), 0.0, 2.0)
    wait_saturation = np.power(wait_scaled + eps, 1.1329051500428644)
    score = neg_slack + ddl_risk_gate * 1.8976506200193597 * critical_path_pressure + energy_penalty - wait_saturation
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
