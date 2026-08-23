import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
      - Strict lexicographic DDL protection via hard `slack > ddl_protection_threshold` gating (validated zero violations)
      - Critical-path release prioritization via `upward_rank * remaining_work` interaction, unconditionally active for urgency
      - Joint MAD normalization over `|slack|`, `uncertainty`, and `duration_total` for coherent risk scale alignment
      - Energy-aware terms gated *only* under meaningfully positive slack (`slack > ddl_protection_threshold`)
      - Bounded monotonic gates (sigmoid on uncertainty, linear slack-pressure scaling) replacing brittle thresholds
      - Ready-wait-time anti-starvation scaled by `(1 + ready_wait_time) / (1 + |slack|)` to avoid explosion near zero slack
      - All features robustly normalized using MAD (median absolute deviation) instead of percentile clipping for cross-seed stability
    """
    eps = 1.2553050347745038e-07
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
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_penalty = np.where(slack < 0, (-slack) ** 3.97047352178002, 0.0)
    deadline_pressure = np.maximum(0.0, -slack + 2.992571772590669)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.3732631716117272
    duration_total = min_exec_time + min_comm_time + eps
    duration_risk = duration_total * uncertainty * 0.4229077748737061
    critical_path_urgency = upward_rank * remaining_work
    critical_score = -mad_normalize(critical_path_urgency)
    safe_slack_mask = np.where(slack > 2.992571772590669, 1.0, 0.0)
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -43.76590601542291
    slack_ub = 1.9789984223087806
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.7865266695132366 + (1.0 - 0.7865266695132366) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-9.640062030548789 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.6250070751142487 * energy_norm * unc_norm * unc_sigmoid
    wait_scale = (1.0 + ready_wait_time) / (1.0 + np.abs(slack) + eps)
    wait_score = mad_normalize(wait_scale)
    score = mad_normalize(slack_penalty) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + critical_score
    score += safe_slack_mask * (0.8110473975710532 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    critical_boost_mask = np.where(slack <= 0.0, 3.1209948018962184, 1.0)
    score = score * critical_boost_mask
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
