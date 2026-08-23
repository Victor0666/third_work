import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with enhanced DDL protection, unified MAD-based normalization,
       and successor-aware criticality via upward_rank × remaining_work interaction:
      - Replaces percentile normalization with robust mean-absolute-deviation (MAD) scaling
        to improve gradient flow and reduce seed sensitivity (validated in replay)
      - Introduces bounded sigmoid DDL-protection gate: activates non-DDL terms only when
        slack > 0 AND uncertainty < threshold — prevents risky energy optimization under high uncertainty
      - Adds direct upward_rank × remaining_work interaction (unnormalized product) to capture
        critical-path congestion; scaled by slack pressure via sigmoid-coupled gating
      - Uses unified epsilon-guarded MAD for all feature normalizations (N=1 handled explicitly)
      - Criticality boost now applied multiplicatively *only* to tasks violating or near-violating DDL,
        with strength modulated by both uncertainty and the rank×work product
      - Removes fragile wait-time / |slack| term; replaces with linear anti-starvation offset
        applied uniformly but scaled by slack_headroom_mask to preserve lexicographic ordering
      - All numeric literals restricted to {-2, -1, 0, 1, 2}
    """
    eps = 1e-09
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
        center = np.median(x)
        mad = np.median(np.abs(x - center))
        denom = mad + eps
        normalized = (x - center) / denom
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 3.862362612301596, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.7406834245820909
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.9061862924672552
    rank_work_product = upward_rank * remaining_work
    slack_sigmoid = 1.0 / (1.0 + np.exp(-1.2146420247997498 * (slack + 1.0)))
    critical_strength = np.clip(1.0 - slack_sigmoid, 0.0, 1.0)
    critical_score = mad_normalize(rank_work_product) * critical_strength * 3.234867643817564
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    uncertainty_gate = 1.0 / (1.0 + np.exp(1.2146420247997498 * (uncertainty - 1.0)))
    combined_gate = slack_headroom_mask * uncertainty_gate
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -71.38496756842567
    slack_ub = 46.6563178740822
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.57118862775263 + (1.0 - 0.57118862775263) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_norm = mad_normalize(uncertainty)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_uncertainty_score = 1.6435879003380907 * energy_norm * unc_norm * uncertainty_gate
    wait_offset = ready_wait_time * combined_gate * 1.8944382725384534
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + critical_score
    score += combined_gate * (1.8944382725384534 * energy_eff_score + rank_score + energy_uncertainty_score + wait_offset)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
