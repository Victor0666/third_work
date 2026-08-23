import numpy as np
RULE_METADATA = {'structure_hash': 'f7e68604782f097b601bc7d8aade239cbf5c3af56b1227c27b43748af794cddc', 'parameter_schema_hash': '7bde2cd8502d222a1e960168da9046115ad25206b8b10d86ef672a308e231721', 'best_parameter_hash': 'e9015ff7fb71a22b13ef02b11d0158fbf19d58ff0e15f44bb386806c2579aa60', 'best_parameters': {'epsilon': 0.002439877520228879, 'slack_risk_penalty': 9.94402013535153, 'slack_urgency_scale': 2.604292099450338, 'energy_efficiency_bias': 2.948984607711699, 'critical_path_leverage': 2.174392021821147, 'wait_fairness_gain': 0.7921565151806211, 'uncertainty_sensitivity': 0.3698257753343346, 'duration_balance': 0.4860113144371899, 'slack_pressure_tanh_scale': 1.384296700543017, 'robustness_mad_factor': 1.0810537370973121}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '417bfb75840e25120e2ea24c063b125e1cfedce3239e733f2c61f884e8c6ab30', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule integrating successor-release interaction and host-load conditional gating.
    
    Key structural mutations:
    - Replaces piecewise slack transformation with smooth tanh-scaled urgency + linear risk penalty
      for unified DDL pressure modeling (no discontinuities at zero)
    - Adds successor-release interaction: upward_rank * remaining_work, gated by slack feasibility
      to unblock critical paths early (counterfactual evidence: 'successor_release_interaction')
    - Introduces host-load conditional gate: energy term is suppressed only when both slack > 0 AND 
      uncertainty <= median_uncertainty, preventing premature energy optimization under risk
    - Replaces wait fairness with saturating exponential: 1 - exp(-gain * norm_wait), bounded and robust
    - Uses MAD-normalized features throughout; all operations epsilon-guarded and inf-safe
    """
    eps = 0.002439877520228879
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.0810537370973121 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    slack_urgency = np.tanh(slack * 1.384296700543017)
    slack_risk = 9.94402013535153 * slack_pressure
    slack_norm = slack_risk + 2.604292099450338 * (1.0 - slack_urgency)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    median_unc = np.median(uncertainty)
    energy_suppression_gate = ((slack > 0.0) & (uncertainty <= median_unc + eps)).astype(float)
    energy_weight_adj = 2.948984607711699 * (1.0 - energy_suppression_gate)
    rank_work_interaction = upward_rank * remaining_work
    rank_work_norm = mad_normalize(rank_work_interaction)
    successor_gate = (slack >= 0.0).astype(float)
    successor_boost = 2.174392021821147 * rank_work_norm * successor_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 0.7921565151806211 * (1.0 - np.exp(-0.7921565151806211 * wait_norm))
    unc_norm = mad_normalize(uncertainty)
    uncertainty_amplifier = 0.3698257753343346 * unc_norm * slack_pressure
    score = slack_norm + 0.4860113144371899 * duration_norm + energy_weight_adj * energy_norm - successor_boost + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
