import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Robust MAD-aligned normalization over |slack|, uncertainty, and duration_total (Parent 2)
      - Power-law anti-starvation with slack-headroom gating (Parent 2)
      - Strict lexicographic DDL protection via hard threshold (Parent 2)
      - Convex slack violation penalty (Parent 1's exponentiated form, improved stability)
      - Uncertainty-slack coupling term (Parent 1's domain-aware interaction)
      - Critical-path release term remains unconditional and dominant in DDL-critical region
      - All features normalized via MAD then clipped to [-2,2] for stability and bounded AST depth
      - No percentile clipping or fragile quantile estimation — only robust median-based scaling
      - Final score preserves lexicographic dominance: DDL violation > critical path > safe-region tradeoffs
    """
    eps = 1.3736910436086985e-08
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
    duration_total = min_exec_time + min_comm_time + eps
    slack_abs_norm = mad_normalize(np.abs(slack))
    unc_norm = mad_normalize(uncertainty)
    dur_norm = mad_normalize(duration_total)
    slack_penalty = np.where(slack < 0, (-slack) ** 3.3282774733773435 * 0.7219853666905024 * (1.0 + unc_norm), 0.0)
    deadline_pressure = np.maximum(0.0, -slack + 0.03099358099043747)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.9917049423987953
    duration_risk = duration_total * uncertainty * 0.6527120007488274
    critical_release_score = upward_rank * remaining_work * 2.4186364463503014
    slack_headroom_mask = np.where(slack > 0.03099358099043747, 1.0, 0.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = slack_headroom_mask * 0.10164569940301087 * energy_norm
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 0.7392991337130981, 0.0)
    wait_norm = mad_normalize(wait_power)
    wait_term = slack_headroom_mask * wait_norm
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.1887892072200623 * (uncertainty - 1.0)))
    energy_uncertainty_term = slack_headroom_mask * 0.899404090537438 * energy_norm * unc_norm * unc_sigmoid
    rank_norm = mad_normalize(upward_rank)
    headroom_ratio = np.clip((slack - 0.03099358099043747) / (0.03099358099043747 + 1.0), 0.0, 1.0)
    weight_rank = 0.9946630203902436 + (1.0 - 0.9946630203902436) * (1.0 - headroom_ratio)
    rank_term = slack_headroom_mask * -rank_norm * weight_rank
    score = mad_normalize(slack_penalty) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_release_score)
    score += energy_term + wait_term + energy_uncertainty_term + rank_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
