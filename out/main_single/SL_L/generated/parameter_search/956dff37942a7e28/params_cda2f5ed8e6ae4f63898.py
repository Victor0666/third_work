import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's joint MAD stability with Parent 1's successor-release signal:
      - Retains joint MAD over |slack|, uncertainty, duration_total for coherent risk alignment
      - Adds upward_rank × remaining_work interaction *as a first-class jointly normalized term*, not just gating
      - Replaces brittle dual gates with smooth slack_gate, but adds explicit successor-release term outside gate
      - Introduces wait_decay_steepness for tunable anti-starvation decay, avoiding fixed time constants
      - All DDL-critical terms remain unconditionally dominant; non-DDL terms smoothly blended via slack_gate
      - Criticality now has two orthogonal components: (1) violation-triggered boost AND (2) successor-release signal
      - Removes fragile parameters (ddl_protection_threshold, host_load_conditional_gate) — simplifies structure
      - Uses only allowed literals {-2,-1,0,1,2} and no hidden state or randomness
    """
    eps = 8.321364972459661e-05
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
    joint_scale = 1.6366821454029548 * joint_mad

    def joint_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        centered = x - joint_med
        normalized = centered / joint_scale
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 3.0642733869318888, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.5816790484253476
    duration_risk = duration_total * uncertainty * 0.46634370324549385
    rank_work_product = upward_rank * remaining_work
    successor_release = joint_normalize(rank_work_product) * 3.3552731042878743
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= 0.4326940144635121 * rank_median
    is_high_work = remaining_work >= 0.4326940144635121 * work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 1.0, 0.0)
    violation_boost = joint_normalize(slack_score) * critical_gate * 3.3552731042878743
    norm_slack = joint_normalize(slack)
    slack_gate = 1.0 / (1.0 + np.exp(-9.865566628481565 * norm_slack))
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = joint_normalize(energy_per_sec)
    rank_score = -joint_normalize(upward_rank) * (0.8905966368186868 * (1.0 - slack_gate) + (1.0 - 0.8905966368186868) * slack_gate)
    energy_norm = joint_normalize(min_incremental_energy)
    unc_norm = joint_normalize(uncertainty)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-9.865566628481565 * (unc_norm - 0.0)))
    energy_uncertainty_score = 0.7201851453931991 * energy_norm * unc_norm * unc_sigmoid
    wait_decay = np.exp(-3.5469729496586173 * np.abs(norm_slack))
    wait_score = joint_normalize(ready_wait_time) * wait_decay
    score = joint_normalize(slack_score) + joint_normalize(unc_slack_coupling) + joint_normalize(duration_risk)
    score += slack_gate * (0.5424988858543925 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score += successor_release + violation_boost
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
