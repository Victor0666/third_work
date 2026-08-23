import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's global scaling & robust DDL gate with Parent 1's smooth feasibility interpolation,
       plus novel energy-density penalty — all within strict 12-parameter limit.
    
    Structural improvements:
      - Adds energy-density penalty: `min_incremental_energy / (remaining_work + eps)` normalized globally,
        directly targeting energy-per-computation-unit efficiency — a core driver of total fuzzy energy minimization.
      - Retains unified global MAD-based feature scaling (Parent 2) for magnitude coherence across heterogeneous units.
      - Keeps simplified DDL-protection via hard slack threshold (Parent 2), but couples critical boost with global-normalized rank.
      - Replaces binary wait-gating with sigmoid using fixed steepness (5.3 → literal 5, allowed as ≤2? no → use 1 instead),
        but since 5 violates literal constraint, we use PARAMS["ddl_protection_threshold"]-derived logic instead:
        wait_sigmoid_gate is replaced by smooth slack-scaled gate: `(1.0 + slack_scaled) / 2.0` — bounded [0.5,1.0], uses only {0,1,2}.
      - All nonlinearities bounded, divisions guarded, and outputs finite with shape (N,).
    """
    eps = 4.2750294777492597e-07
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
    all_features = np.stack([min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty], axis=0).flatten()
    global_med = np.median(all_features) if len(all_features) > 1 else np.mean(all_features)
    global_mad = np.median(np.abs(all_features - global_med)) if len(all_features) > 1 else eps
    global_scale = 1.4404806243115602 * (global_mad if global_mad > eps else eps)

    def global_normalize(x):
        x = np.asarray(x, dtype=float)
        return np.clip((x - global_med) / global_scale, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.399550639666373, 0.0)
    slack_lb = -27.060572685848015
    slack_ub = 21.51037921732383
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    ddl_protection_gate = (slack <= -5.379648770028887).astype(float)
    critical_boost = ddl_protection_gate * 1.752036515752363 * global_normalize(upward_rank)
    wait_sigmoid_gate = (1.0 + slack_scaled) / 2.0
    wait_normalized = ready_wait_time / (np.abs(slack) + 1.0)
    wait_score = global_normalize(wait_normalized) * wait_sigmoid_gate
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_norm = global_normalize(energy_per_sec)
    energy_density = min_incremental_energy / (remaining_work + eps)
    energy_density_norm = global_normalize(energy_density)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.6010652709967754
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.8394755883858714
    weight_rank = 0.27863021644995356 * (1.0 - slack_scaled)
    rank_norm = global_normalize(upward_rank)
    rank_score = -rank_norm * weight_rank
    median_unc = np.median(uncertainty) if N > 1 else np.mean(uncertainty)
    unc_threshold_gate = (uncertainty <= median_unc + eps).astype(float)
    energy_norm = global_normalize(min_incremental_energy)
    unc_norm = global_normalize(uncertainty)
    energy_uncertainty_score = 1.1707851392289996 * energy_norm * unc_norm * unc_threshold_gate
    score = global_normalize(slack_score) + global_normalize(unc_slack_coupling) + global_normalize(duration_risk) + 0.6197678448984304 * energy_eff_norm + 1.0 * energy_density_norm + rank_score + energy_uncertainty_score + wait_score - critical_boost
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
