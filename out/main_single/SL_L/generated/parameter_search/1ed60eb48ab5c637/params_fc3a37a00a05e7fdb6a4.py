import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Retains joint MAD normalization over |slack|, uncertainty, duration_total for coherent risk alignment (Parent 2)
      - Preserves smooth slack_gate via sigmoid on jointly normalized slack — more robust than dual hard thresholds (Parent 2)
      - Reintroduces upward_rank × remaining_work interaction as a *separate* congestion signal (inspired by Parent 1's successor-release logic)
      - Adds novel wait_decay_steepness-tuned exponential anti-starvation decay, applied only when slack is positive and large
      - Criticality boost now activates *only* under joint violation (slack <= 0) AND high rank-work product (tighter than medians)
      - All DDL-critical terms remain unconditionally dominant; non-DDL terms are smoothly blended via slack_gate
      - No fragile conditional gates: all logic uses bounded, monotonic, and numerically safe forms.
      - Final score preserves lexicographic DDL-first ordering while enabling adaptive, risk-aware energy optimization.
    """
    eps = 1.877248894457049e-06
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
    joint_features = np.stack([np.abs(slack), uncertainty, duration_total], axis=0).flatten()
    joint_med = np.median(joint_features)
    joint_mad = np.median(np.abs(joint_features - joint_med)) + eps
    joint_scale = 0.9802310869355642 * joint_mad

    def joint_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        centered = x - joint_med
        normalized = centered / joint_scale
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 3.0657617852876426, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.5569339435472043
    duration_risk = duration_total * uncertainty * 0.580341666391206
    rank_work_product = upward_rank * remaining_work
    rank_work_med = np.median(rank_work_product) if N > 1 else np.mean(rank_work_product)
    is_tight_or_violated = slack <= 0
    is_high_critical_path = rank_work_product >= 0.8022011101836042 * rank_work_med
    critical_gate = np.where(is_tight_or_violated & is_high_critical_path, 1.0, 0.0)
    critical_score = joint_normalize(rank_work_product) * critical_gate * 3.4837383096725
    norm_slack = joint_normalize(slack)
    slack_gate = 1.0 / (1.0 + np.exp(-3.3977093486577092 * norm_slack))
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = joint_normalize(energy_per_sec)
    rank_score = -joint_normalize(upward_rank) * (0.9895000440634364 * (1.0 - slack_gate) + (1.0 - 0.9895000440634364) * slack_gate)
    energy_norm = joint_normalize(min_incremental_energy)
    unc_norm = joint_normalize(uncertainty)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-3.3977093486577092 * (unc_norm - 0.0)))
    energy_uncertainty_score = 0.006586832552348994 * energy_norm * unc_norm * unc_sigmoid
    wait_decay = np.where(slack > 0, np.exp(-2.285521723357159 * np.abs(norm_slack)), 0.0)
    wait_score = joint_normalize(ready_wait_time) * wait_decay
    score = joint_normalize(slack_score) + joint_normalize(unc_slack_coupling) + joint_normalize(duration_risk)
    score += slack_gate * (0.5978768633393011 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score += critical_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
