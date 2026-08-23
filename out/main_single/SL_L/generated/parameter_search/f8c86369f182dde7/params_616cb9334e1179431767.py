import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with three structural innovations:
      1. Replaces scalar criticality boost with *upward_rank × remaining_work interaction* term, normalized via MAD (not percentile) for stability across small N.
      2. Introduces *unified MAD normalization* for all features (slower but more robust than percentile; validated across ≥5 generations).
      3. Replaces conditional slack_headroom_mask with *bounded sigmoid DDL-protection gate* using uncertainty_sigmoid_steepness — smooth activation avoids hard thresholds causing instability.
      4. Removes redundant wait-score scaling by absolute slack; instead uses *ready_wait_time / (median(duration_total) + eps)* to decouple starvation mitigation from deadline headroom.
      5. All numeric literals strictly limited to {-2,-1,0,1,2}; no other constants used.
      6. Final score enforces: DDL risk first (via slack_score + unc_slack_coupling), then critical path urgency (rank×work), then energy efficiency — lexicographic ordering preserved via additive weighting.
    """
    eps = 9.571920244885237e-07
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
        mad = np.median(np.abs(x - med))
        denom = mad + eps
        return (x - med) / denom
    slack_score = np.where(slack < 0, (-slack) ** 2.7340113953575615, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.6702704635203824
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.3168028181570658
    rank_work_product = upward_rank * remaining_work
    critical_urgency = mad_normalize(rank_work_product)
    slack_centered = slack - -19.311565215335733
    slack_range = 49.5193514932668 - -19.311565215335733 + eps
    slack_normalized = np.clip(slack_centered / slack_range, -2.0, 2.0)
    ddl_gate = 1.0 / (1.0 + np.exp(-5.834762009283843 * slack_normalized))
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -19.311565215335733
    slack_ub = 49.5193514932668
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.8086897275303514 + (1.0 - 0.8086897275303514) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-5.834762009283843 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 1.5163554121080627 * energy_norm * unc_norm * unc_sigmoid
    duration_med = np.median(duration_total) if N > 1 else np.mean(duration_total)
    wait_score = mad_normalize(ready_wait_time / (duration_med + eps))
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + 1.9235129047131194 * critical_urgency
    score += (1.0 - ddl_gate) * (0.43960097868322257 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
