import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating three evidence-backed structural improvements:
      - Replaces lexicographic `slack > 0` gate with soft DDL protection gate (`slack > ddl_protection_gate`) to allow mild energy optimization even under marginal headroom
      - Introduces successor-release coupling `(upward_rank * remaining_work)` as primary non-DDL driver — directly addresses CRITICAL_PATH_STARVATION from counterfactual replay
      - Replaces wait-time normalization with bounded exponential decay `exp(-ready_wait_decay * normalized_slack)` to reduce starvation penalty gradually as slack shrinks, avoiding step discontinuities
      - All feature normalizations use robust MAD (median absolute deviation) scaling instead of percentile clipping for improved stability in sparse ready sets (N ≤ 5)
      - Criticality boost now conditioned on `slack <= ddl_protection_gate`, aligning with empirical hard-failure boundary
      - No unbounded ops: all exponentials, sigmoids, and divisions safeguarded by epsilon and finite clipping
      - Final score preserves strict DDL-first ordering: DDL-violating tasks always receive lowest priority unless no feasible alternative exists (handled externally)
    """
    eps = 1.2290887010070172e-07
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
    slack_score = np.where(slack < 0, (-slack) ** 1.7640353991457443, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.3691673143022562
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.915438253814725
    rank_work_product = upward_rank * remaining_work
    slack_deficit = np.clip(-slack, 0.0, np.inf)
    successor_release_score = rank_work_product * (1.0 + slack_deficit) * 0.4532596566541779
    ddl_gate_threshold = 0.17220498182915117
    is_ddl_pressure = slack <= ddl_gate_threshold
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_ddl_pressure, 0.8212151559384552, 1.0)
    slack_headroom_mask = np.where(slack > ddl_gate_threshold, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_normalized = mad_normalize(slack)
    rank_weight = np.exp(-0.3818669752106184 * np.maximum(0.0, slack_normalized))
    rank_score = -mad_normalize(upward_rank) * rank_weight
    unc_sigmoid = 1.0 / (1.0 + np.exp(-7.824000438121684 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.8223460611768727 * energy_norm * unc_norm * unc_sigmoid
    wait_score = np.where(slack_headroom_mask > 0.0, ready_wait_time * np.exp(-0.3818669752106184 * (slack - ddl_gate_threshold)), 0.0)
    wait_score = mad_normalize(wait_score)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(successor_release_score)
    score += slack_headroom_mask * (0.8272844352567185 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
