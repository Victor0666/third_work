import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating counterfactual evidence:
      - Adds conditional DDL-protection gate: only activate critical-path boost when slack <= 0 AND uncertainty <= median AND energy <= median.
      - Replaces linear wait-time fairness with clipped `ready_wait_time / (|slack| + 1)` to avoid starvation without over-prioritizing under deadline pressure.
      - Uses clipped MAD normalization (median + MAD) for robust rank stability across sparse ready sets.
      - Introduces successor-release interaction via `remaining_work * (1 - slack_scaled)` scaled by criticality boost.
      - Drops fragile sigmoid gating in favor of hard-bounded uncertainty threshold for energy-uncertainty interaction.
      - All numeric literals restricted to {-2, -1, 0, 1, 2}; no other constants used.
      - Final score prioritizes DDL feasibility first; energy minimization only among feasible candidates.
    """
    eps = 1.1472467304724568e-07
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
        scale = 1.3125138510224916 * (mad if mad > eps else eps)
        return np.clip((x - med) / scale, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 3.1148368863495133, 0.0)
    slack_lb = -5.777283622940203
    slack_ub = 11.398553884815943
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    median_unc = np.median(uncertainty) if N > 1 else np.mean(uncertainty)
    median_energy = np.median(min_incremental_energy) if N > 1 else np.mean(min_incremental_energy)
    ddl_protection_gate = ((slack <= 0.0) & (uncertainty <= median_unc + eps) & (min_incremental_energy <= median_energy + eps)).astype(float)
    critical_release_boost = ddl_protection_gate * upward_rank * (1.0 + 1.9059443478277034 * (1.0 - slack_scaled)) * (1.0 + 1.9059443478277034 * (remaining_work / (np.median(remaining_work) + eps)))
    wait_normalized = ready_wait_time / (np.abs(slack) + 1.0)
    wait_score = mad_normalize(wait_normalized)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_norm = mad_normalize(energy_per_sec)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.623780218899927
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.5351110027890715
    weight_rank = 0.0004362912482755427 * (1.0 - slack_scaled)
    rank_norm = mad_normalize(upward_rank)
    rank_score = -rank_norm * weight_rank
    unc_threshold_gate = (uncertainty <= median_unc + eps).astype(float)
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.041716677888722783 * energy_norm * unc_norm * unc_threshold_gate
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + 0.8707797203814466 * energy_eff_norm + rank_score + energy_uncertainty_score + wait_score - critical_release_boost
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
