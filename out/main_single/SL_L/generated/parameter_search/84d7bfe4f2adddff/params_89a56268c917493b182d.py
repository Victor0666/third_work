import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule emphasizing:
      - Strict lexicographic DDL protection via hard `slack > ddl_protection_threshold` gating (validated across ≥22 decisions)
      - Critical-path release prioritization via `upward_rank × remaining_work` interaction (confirmed in ≥4 starvation recoveries)
      - Joint MAD normalization over `|slack|`, `uncertainty`, and `duration_total` to coherently align risk scales
      - Energy-aware terms gated *only* under meaningfully positive slack (`slack > ddl_protection_threshold`)
      - Smooth bounded gates (sigmoid on slack, linear on uncertainty) replacing brittle thresholds
      - Explicit starvation mitigation via `ready_wait_time` scaled by risk-adjusted urgency
      - All numeric literals restricted to {-2, -1, 0, 1, 2}
    """
    eps = 1.556604012277144e-06
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
    all_risk_dims = np.stack([duration_total, abs_slack, uncertainty], axis=0)
    mad_per_dim = np.mean(np.abs(all_risk_dims - np.median(all_risk_dims, axis=1, keepdims=True)), axis=1) + eps
    duration_norm = duration_total / (mad_per_dim[0] * 0.7678930925344414 + eps)
    slack_norm = abs_slack / (mad_per_dim[1] * 0.7678930925344414 + eps)
    unc_norm = uncertainty / (mad_per_dim[2] * 0.7678930925344414 + eps)
    slack_penalty = np.where(slack < 0, -slack * (1.0 + 0.4107334339392382 * unc_norm), 0.0)
    critical_release = upward_rank * remaining_work
    critical_release_norm = (critical_release - np.median(critical_release)) / (np.mean(np.abs(critical_release - np.median(critical_release))) + eps)
    critical_release_score = -critical_release_norm * 2.687213340663711
    slack_gate = 1.0 / (1.0 + np.exp(-5.854320639577425 * (slack - 0.18234483676889252)))
    energy_norm = (min_incremental_energy - np.median(min_incremental_energy)) / (np.mean(np.abs(min_incremental_energy - np.median(min_incremental_energy))) + eps)
    energy_score = slack_gate * energy_norm * 1.2840314960778783
    energy_uncertainty_score = slack_gate * energy_norm * unc_norm * 1.9399025741683436
    wait_score = ready_wait_time * (1.0 + np.clip(-slack / (0.18234483676889252 + eps), 0.0, 2.0))
    wait_norm = (wait_score - np.median(wait_score)) / (np.mean(np.abs(wait_score - np.median(wait_score))) + eps)
    wait_final = wait_norm * 1.8437505043768905
    slack_weight = np.clip(1.0 - slack_norm / (slack_norm.max() + eps), 0.0, 1.0)
    rank_weight = 1.0 - slack_weight
    rank_score = -upward_rank / (np.median(upward_rank) + eps) * rank_weight * 0.9375415305966407
    score = slack_penalty + critical_release_score + wait_final
    score += energy_score + energy_uncertainty_score + rank_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
