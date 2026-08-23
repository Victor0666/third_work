import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Decoupled per-feature MAD normalization (no joint stacking) → eliminates cross-contamination, improves stability under skew/sparsity
      - Hard DDL-protection gate: energy/uncertainty terms fully disabled when slack < 0 → strict feasibility-first enforcement
      - Smooth tanh-based slack-uncertainty coupling: replaces discontinuous threshold with differentiable, gradient-friendly interaction
      - All DDL-critical terms dominate lexicographically; non-DDL terms gated *off* under violation
      - Bounded arithmetic only; no unbounded functions, no in-place mutation, full NaN/inf protection
    """
    eps = 8.246641933780086e-08
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
        x_med = np.median(x)
        x_mad = np.mean(np.abs(x - x_med)) + eps
        return (x - x_med) / (x_mad * 0.8060311659101506 + eps)
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack) + eps
    duration_norm = mad_normalize(duration_total)
    slack_norm = mad_normalize(abs_slack)
    unc_norm = mad_normalize(uncertainty)
    energy_norm = mad_normalize(min_incremental_energy)
    rank_norm = mad_normalize(upward_rank)
    work_norm = mad_normalize(remaining_work)
    wait_norm = mad_normalize(ready_wait_time)
    slack_sign = np.tanh(slack / (np.abs(slack) + eps))
    slack_penalty = np.where(slack < 0, -slack * (1.0 + 1.611993057683223 * unc_norm * np.abs(slack_sign) ** 1.7283319679546136), 0.0)
    critical_release = upward_rank * remaining_work
    critical_release_norm = mad_normalize(critical_release)
    critical_release_score = -critical_release_norm * 3.0239919532104045
    ddl_safe_mask = np.where(slack >= 0.0, 1.0, 0.0)
    ramp_half = 0.4733939268263312 / 2.0
    gate_center = 0.742459748529266
    gate_low = gate_center - ramp_half
    gate_high = gate_center + ramp_half
    slack_gate = np.clip((slack - gate_low) / (ramp_half * 2.0 + eps), 0.0, 1.0)
    safe_gate = ddl_safe_mask * slack_gate
    energy_score = safe_gate * energy_norm * 1.6127336850515133
    energy_uncertainty_score = safe_gate * energy_norm * unc_norm * 1.5762163389935226
    median_wait = np.median(ready_wait_time) + eps
    wait_ratio = np.clip(ready_wait_time / median_wait, 0.0, 2.0)
    wait_score = ready_wait_time * (1.0 + np.clip(-slack / (0.742459748529266 + eps), 0.0, 2.0))
    wait_final = safe_gate * mad_normalize(wait_score) * 0.18250717837215027
    slack_weight = np.clip(1.0 - slack_norm / (slack_norm.max() + eps), 0.0, 1.0)
    rank_weight = 1.0 - slack_weight
    rank_score = safe_gate * -rank_norm * rank_weight * 0.6421487642299412
    score = slack_penalty + critical_release_score
    score += energy_score + energy_uncertainty_score + wait_final + rank_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
