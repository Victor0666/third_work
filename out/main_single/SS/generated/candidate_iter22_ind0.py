import numpy as np
RULE_METADATA = {'structure_hash': '75ffe5ed90f865c099c4f61656800bac4d4637edc714c4e99198e868c4128982', 'parameter_schema_hash': '4a74dd1f213f2b4ef916ccc9c08e835b373bb84bbd90a97fb6de2c3ccfed89b8', 'best_parameter_hash': '6bef26765cd782af34022f0af37f739c363ca6a95bc05ce11bf135e758ef11f2', 'best_parameters': {'epsilon': 0.00010127205696678124, 'slack_penalty_exponent': 2.0744686260358662, 'criticality_scale': 1.0403034969180864, 'energy_sensitivity': 0.37656058966330563, 'energy_uncertainty_interaction': 0.7900797741165348, 'ddl_protection_gate_slope': 6.10615530673001, 'remaining_work_weight': 0.7327621594460204, 'slack_pressure_gate_steepness': 2.168381722791708, 'uncertainty_gate_threshold': 0.0036939494417571666, 'wait_decay': 0.3123109922539546, 'quantile_q1': 0.14955103644588333, 'quantile_q3': 0.7488972994651452}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '4e4c188296928cc4f0239c95756c0b3a6fafb90312b621150bc24ed72dac4007', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces median-MAD with quantile-based robust normalization (Q1/Q3);
       introduces joint congestion signal (ready_wait_time + uncertainty) gated by both DDL feasibility and high-congestion threshold;
       removes redundant duration_robustness and wait_saturation_offset; replaces coupled_slack with sign-preserving MAD-normalized slack;
       adds conditional ddl_protection_gate activated only for slack >= 0 to isolate energy/uncertainty penalties to feasible region;
       enforces critical-path awareness via upward_rank × remaining_work × ddl_gate only under deadline pressure."""
    eps = 0.00010127205696678124
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def quantile_normalize(x):
        x = np.copy(x)
        if N == 1:
            q1 = q3 = x[0]
            iqr = eps
        else:
            q1 = np.quantile(x, 0.14955103644588333)
            q3 = np.quantile(x, 0.7488972994651452)
            iqr = q3 - q1
        spread = iqr if iqr > eps else eps
        return (x - q1) / spread
    norm_slack = quantile_normalize(slack)
    norm_energy = quantile_normalize(min_incremental_energy)
    norm_duration = quantile_normalize(min_exec_time + min_comm_time)
    norm_rank = quantile_normalize(upward_rank)
    norm_work = quantile_normalize(remaining_work)
    norm_wait = quantile_normalize(ready_wait_time)
    norm_uncert = quantile_normalize(uncertainty)
    slack_mad = np.median(np.abs(slack - np.median(slack)))
    slack_mad = max(slack_mad, eps)
    norm_slack_sharp = (slack - np.median(slack)) / slack_mad
    ddl_feasible = (slack >= 0.0).astype(float)
    ddl_protection_gate = 1.0 / (1.0 + np.exp(-6.10615530673001 * (slack + eps)))
    critical_path_leverage = norm_rank * norm_work * ddl_feasible
    raw_neg_slack = np.maximum(-slack, 0.0)
    slack_penalty = raw_neg_slack ** 2.0744686260358662
    norm_slack_penalty = quantile_normalize(slack_penalty)
    congestion = ready_wait_time + uncertainty
    congestion_norm = quantile_normalize(congestion)
    congestion_gate = (congestion_norm > 0.0036939494417571666).astype(float)
    congestion_penalty = congestion_norm * congestion_gate * ddl_feasible
    wait_benefit = np.clip(0.3123109922539546 * ready_wait_time, 0.0, 2.0)
    energy_uncert_interaction = norm_energy * norm_uncert * congestion_gate * ddl_feasible
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(norm_rank, -2.0, 2.0) * (1.0 + 1.0403034969180864 * np.clip(-norm_slack_sharp, 0.0, 1.0)) - 0.37656058966330563 * np.clip(norm_energy * ddl_protection_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(congestion_penalty, -2.0, 2.0) + 0.7900797741165348 * np.clip(energy_uncert_interaction, -2.0, 2.0) + 0.7327621594460204 * np.clip(norm_work, -2.0, 2.0) + np.clip(2.168381722791708 * np.clip(-norm_slack_sharp, 0.0, 2.0), -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
