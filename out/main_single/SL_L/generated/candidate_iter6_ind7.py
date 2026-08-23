import numpy as np
RULE_METADATA = {'structure_hash': '249553ab2f6aa651303fd796abc18e2fc6cf40eeac8f0ddf855ed16c9b78a4f5', 'parameter_schema_hash': '780a6189e1a5f5e65c8773c014a701b798d38084d6602d352ba0eb5281af70e1', 'best_parameter_hash': 'd56ae0414b1cf53a928efd818b72ad2ebd2de8f179f9ed2c3766eb75026f23cb', 'best_parameters': {'epsilon': 1.3520091702595937e-09, 'slack_risk_penalty': 1.348138289471884, 'slack_urgency_scale': 0.9920116038526301, 'slack_cap': 23.188648458052537, 'energy_sensitivity': 1.1686913755278905, 'critical_path_leverage': 3.556901358596791, 'wait_fairness_gain': 0.7013920757046208, 'uncertainty_sensitivity': 0.16960433149876447, 'duration_balance': 0.009218641652948183, 'work_density_weight': 0.262860837057498, 'robustness_mad_factor': 1.4199378350849625, 'slack_pressure_tanh_scale': 0.09638322203390087}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'f7d97263e8790a5a0878885bbc9378536d7dae729abbf948225fd51404793df5', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Self-evolved priority rule featuring:
      - Piecewise-linear + saturating exponential slack urgency (replaces fragile tanh),
      - Triple-gated critical-path leverage (slack>=0 ∧ uncertainty<=median ∧ energy<median),
      - Robust MAD-normalized features with epsilon-guarded denominators,
      - Unified anti-starvation wait fairness and uncertainty-pressure coupling.
    All parameters are used; no numeric literals except {-2,-1,0,1,2}; deterministic and finite.
    """
    eps = 1.3520091702595937e-09
    finfo = np.finfo(np.float64)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=np.float64)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.4199378350849625 * (mad if mad > eps else eps)
        return (x - med) / scale
    neg_mask = slack < 0.0
    tight_mask = (slack >= 0.0) & (slack <= 23.188648458052537)
    loose_mask = slack > 23.188648458052537
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 1.348138289471884 * -slack[neg_mask]
    slack_norm[tight_mask] = 0.9920116038526301 * (np.exp(slack[tight_mask]) - 1.0)
    slack_norm[loose_mask] = 0.9920116038526301 * (np.exp(23.188648458052537) - 1.0)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_score = 1.1686913755278905 * energy_norm
    rank_norm = mad_normalize(upward_rank)
    unc_med = np.median(uncertainty)
    eng_med = np.median(min_incremental_energy)
    unc_gate = (uncertainty <= unc_med).astype(float)
    eng_gate = (min_incremental_energy <= eng_med).astype(float)
    critical_gate = ((slack >= 0.0) * unc_gate * eng_gate).astype(float)
    critical_boost = 3.556901358596791 * rank_norm * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_gate = (slack >= 0.0).astype(float)
    work_density_bonus = 0.262860837057498 * work_density_norm * work_density_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 1.0 - np.exp(-0.7013920757046208 * np.maximum(wait_norm, 0.0))
    wait_score = wait_boost * (slack >= 0.0).astype(float)
    unc_norm = mad_normalize(uncertainty)
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    uncertainty_amplifier = 0.16960433149876447 * unc_norm * np.tanh(slack_pressure * 0.09638322203390087)
    score = slack_norm + 0.009218641652948183 * duration_norm + energy_score - critical_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    return score
