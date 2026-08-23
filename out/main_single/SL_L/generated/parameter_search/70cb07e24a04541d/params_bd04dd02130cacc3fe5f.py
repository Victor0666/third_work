import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's robustness with Parent 1's anti-starvation expressiveness:
      - Retains Parent 2's bounded duration risk, slack-sigmoid gating, and direct slack-rank coupling.
      - Integrates Parent 1's power-law wait-time boost but gated by slack headroom (not binary), enhancing fairness without compromising DDL safety.
      - Uses joint MAD normalization across all risk dimensions (duration, |slack|, uncertainty) for coherent scaling.
      - All operations guarded against NaN/inf; no in-place mutation; deterministic finite output.
      - Final score structure: [DDL violation] + [tight-slack risk] + [critical path release] + [starvation mitigation (power-law + headroom)] + [energy-uncertainty interaction (smoothly gated)].
    """
    eps = 1.108077776056574e-07
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
    duration_norm = duration_total / (mad_per_dim[0] * 1.9862372610587955 + eps)
    slack_norm = abs_slack / (mad_per_dim[1] * 1.9862372610587955 + eps)
    unc_norm = uncertainty / (mad_per_dim[2] * 1.9862372610587955 + eps)
    slack_penalty = np.where(slack < 0, -slack * (1.0 + 0.3118130050529132 * unc_norm), 0.0)
    duration_risk_base = (min_exec_time + min_comm_time) * uncertainty
    duration_risk_penalty = np.where(slack <= 0.14520043831510543, duration_risk_base * 0.42476511497658254, 0.0)
    critical_release = upward_rank * remaining_work
    critical_release_median = np.median(critical_release)
    critical_release_mad = np.mean(np.abs(critical_release - critical_release_median)) + eps
    critical_release_norm = (critical_release - critical_release_median) / critical_release_mad
    critical_release_score = -critical_release_norm * 1.5787100715821754
    slack_gate = 1.0 / (1.0 + np.exp(-8.610584498665075 * (slack - 0.14520043831510543)))
    energy_median = np.median(min_incremental_energy)
    energy_mad = np.mean(np.abs(min_incremental_energy - energy_median)) + eps
    energy_norm = (min_incremental_energy - energy_median) / (energy_mad + eps)
    energy_score = slack_gate * energy_norm * 0.10584858433638496
    energy_uncertainty_score = slack_gate * energy_norm * unc_norm * 0.6681984916748032
    slack_headroom = np.clip(slack - 0.14520043831510543, 0.0, np.inf)
    wait_power = ready_wait_time ** 1.3185325967547725
    wait_score = wait_power * (1.0 + slack_headroom / (0.14520043831510543 + eps))
    wait_median = np.median(wait_score)
    wait_mad = np.mean(np.abs(wait_score - wait_median)) + eps
    wait_norm = (wait_score - wait_median) / wait_mad
    wait_final = wait_norm * 0.05215016561537162
    slack_distance = np.clip(0.14520043831510543 - slack, 0.0, np.inf)
    slack_rank_score = -upward_rank * (1.0 + slack_distance / (0.14520043831510543 + eps)) * 1.9420859379962554
    score = slack_penalty + duration_risk_penalty + critical_release_score + wait_final + slack_rank_score
    score += energy_score + energy_uncertainty_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
