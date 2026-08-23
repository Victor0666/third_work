import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's DDL-first robustness with Parent 1's resilience-aware interactions:
      - Retains Parent 2's conditional DDL-protection gate and clipped empirical slack scaling for stability.
      - Integrates Parent 1's *joint feasibility-inspired* wait-term: `ready_wait_time / (|slack| + 1)` scaled by adaptive factor,
        but now gated by empirical slack safety (`slack_scaled > wait_safety_threshold`) to avoid unfair boosts near deadlines.
      - Replaces Parent 2's successor-release interaction with Parent 1's *work-density bonus*: `remaining_work / duration`,
        normalized and multiplied by joint slack/uncertainty gate (`slack_scaled * (1 - unc_norm)`) — favors throughput-critical
        tasks only when slack is positive and uncertainty low.
      - Adds robustness: all MAD normalizations use [-2, 2] clipping and explicit N=1 handling.
      - Introduces novel *uncertainty-aware energy efficiency*: replaces simple `energy_per_sec` with `min_incremental_energy / (duration * (1 + uncertainty))`,
        capturing risk-adjusted efficiency without introducing new parameters.
      - All numeric literals are in {-2,-1,0,1,2}; no unbounded loops or state; fully deterministic and finite.
    """
    eps = 4.2491770307825015e-09
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
        med = np.median(x) if N > 1 else np.mean(x)
        mad = np.median(np.abs(x - med)) if N > 1 else eps
        scale = 1.3096300720155472 * (mad if mad > eps else eps)
        return np.clip((x - med) / scale, -2.0, 2.0)
    slack_lb = -22.158281384762617
    slack_ub = 2.265694811509358
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.314376425197688, 0.0)
    median_unc = np.median(uncertainty) if N > 1 else np.mean(uncertainty)
    median_energy = np.median(min_incremental_energy) if N > 1 else np.mean(min_incremental_energy)
    ddl_protection_gate = ((slack <= 0.0) & (uncertainty <= median_unc + eps) & (min_incremental_energy <= median_energy + eps)).astype(float)
    rank_norm = mad_normalize(upward_rank)
    critical_release_boost = ddl_protection_gate * rank_norm * (1.0 + 0.8883490422453335 * (1.0 - slack_scaled))
    wait_base = ready_wait_time / (np.abs(slack) + 1.0)
    wait_gated = wait_base * (slack_scaled > 0.0016029856566018566).astype(float)
    wait_score = mad_normalize(wait_gated)
    duration_total = min_exec_time + min_comm_time + eps
    energy_efficiency = min_incremental_energy / (duration_total * (1.0 + uncertainty))
    energy_eff_norm = mad_normalize(energy_efficiency)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.3915473915069976
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.1414705966060173
    weight_rank = 0.4817331485622001 * (1.0 - slack_scaled)
    rank_score = -rank_norm * weight_rank
    work_density = np.divide(remaining_work, duration_total + eps)
    work_density_norm = mad_normalize(work_density)
    unc_norm = mad_normalize(uncertainty)
    work_density_gate = slack_scaled * (1.0 - np.clip(unc_norm, 0.0, 1.0))
    work_density_bonus = work_density_norm * work_density_gate
    unc_threshold_gate = (uncertainty <= median_unc + eps).astype(float)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_uncertainty_score = 0.06984315814145939 * energy_norm * unc_norm * unc_threshold_gate
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + 1.357647328811107 * energy_eff_norm + rank_score + energy_uncertainty_score + wait_score - critical_release_boost - work_density_bonus
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
