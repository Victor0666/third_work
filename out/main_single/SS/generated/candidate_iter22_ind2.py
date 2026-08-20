import numpy as np
RULE_METADATA = {'structure_hash': 'a93614f4b0435e4d6be62a832987bffa5ddce94dbb571c32593b398a03ff7f3d', 'parameter_schema_hash': '5ac78207d23f51634bde97973f5db0efe53216bb0081e5c0feaca938bd936656', 'best_parameter_hash': 'c4131bb25c98e9caae8d5a4a71706e4dd09f593ae9ed327d0bdd6ddeb086f297', 'best_parameters': {'epsilon': 1.2331249878234342e-06, 'slack_penalty_exponent': 1.4705409032605654, 'energy_sensitivity': 0.10420026972293758, 'energy_uncertainty_interaction': 0.18348374118594646, 'ddl_protection_gate_slope': 5.802133381953219, 'remaining_work_weight': 0.8543243272147536, 'slack_pressure_gate_steepness': 1.0011822673189468, 'uncertainty_gate_threshold': 0.33002934701255693, 'wait_decay': 0.07203626799475835, 'quantile_q1': 0.23658743261624432, 'quantile_q3': 0.5611474213616869}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '1ab52a0277bcfdbb5606b0088523badbe9f8ff1e14e3358a0c32dceb0dd54603', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces median-MAD with quantile-based robust normalization;
       introduces congestion-aware gating via (ready_wait_time + uncertainty) threshold;
       replaces boosted_rank with conditional critical-path coupling (upward_rank × remaining_work) activated only under ddl pressure;
       removes duration_robustness and wait_saturation_offset (deemed inactive per diagnostics);
       enforces sign-preserving slack normalization via MAD;
       adds load-successor-release interaction: penalizes high-uncertainty tasks with large remaining_work when slack is tight."""
    eps = 1.2331249878234342e-06
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
            q1 = x[0]
            q3 = x[0]
        else:
            q1 = np.quantile(x, 0.23658743261624432)
            q3 = np.quantile(x, 0.5611474213616869)
        iqr = q3 - q1
        spread = iqr if iqr > eps else eps
        return (x - q1) / spread

    def mad_normalize_slack(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = mad_normalize_slack(slack)
    norm_energy = quantile_normalize(min_incremental_energy)
    norm_duration = quantile_normalize(min_exec_time + min_comm_time)
    norm_rank = quantile_normalize(upward_rank)
    norm_work = quantile_normalize(remaining_work)
    norm_wait = quantile_normalize(ready_wait_time)
    norm_uncert = quantile_normalize(uncertainty)
    ddl_feasible = (slack > 0.0).astype(float)
    ddl_breach = 1.0 - ddl_feasible
    critical_path_coupling = norm_rank * norm_work * ddl_breach
    ddl_protection_gate = 1.0 / (1.0 + np.exp(-5.802133381953219 * slack))
    congestion = ready_wait_time + uncertainty
    congestion_norm = quantile_normalize(congestion)
    congestion_gate = 1.0 / (1.0 + np.exp(-5.802133381953219 * (congestion_norm - 0.33002934701255693)))
    congestion_penalty = norm_duration * congestion_gate * ddl_feasible
    successor_release_penalty = norm_uncert * norm_work * ddl_breach
    wait_benefit = np.clip(0.07203626799475835 * ready_wait_time, 0.0, 2.0)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.4705409032605654
    norm_slack_penalty = quantile_normalize(raw_slack_penalty)
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) + np.clip(critical_path_coupling, -2.0, 2.0) - np.clip(norm_rank, -2.0, 2.0) - 0.10420026972293758 * np.clip(norm_energy * ddl_protection_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(congestion_penalty, -2.0, 2.0) + 0.18348374118594646 * np.clip(successor_release_penalty, -2.0, 2.0) + 0.8543243272147536 * np.clip(norm_work * ddl_feasible, -2.0, 2.0) + 1.0011822673189468 * np.clip(norm_slack * ddl_feasible, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
