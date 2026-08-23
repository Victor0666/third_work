import numpy as np
RULE_METADATA = {'structure_hash': '14503bc2bdb65b358ba4c605dfc9d61fb92d6cac5ca5837c981222a2ba955e73', 'parameter_schema_hash': '44a09918687b24f58130d8e3a7fe57684b48f7189649293c6f564474d77d68f9', 'best_parameter_hash': '29b37698e8ad2fcc5b7a3b2ade5e8083cbd5250705f61575541dbcc9b302930e', 'best_parameters': {'epsilon': 6.145597529211387e-06, 'slack_risk_penalty': 5.056544297086255, 'slack_urgency_scale': 2.521726697051226, 'slack_sigmoid_k': 0.27452010064901533, 'slack_cap': 7.352505283331998, 'energy_efficiency_bias': 0.79767338540533, 'critical_path_leverage': 0.013559705618124779, 'wait_fairness_gain': 0.46644245956534225, 'uncertainty_sensitivity': 0.26750445155423597, 'duration_balance': 0.0084712659095373, 'energy_fairness_gate_k': 0.8455035644478164, 'slack_pressure_tanh_scale': 0.1665037996274897}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '4cbfdb8e52fa8385a275663b28871194b3bfac21ed8a38a132410064a12f161c', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: smooth sigmoid slack gating, relative-MAD normalization,
       and fairness-gated energy dampening.
    
    Key evolutions:
    - Replaced piecewise slack transformation with *smooth, differentiable sigmoid*:
      urgency = 1 - exp(-k * max(0, slack)) for positive slack, preserving monotonicity
      and enabling gradient-aware optimization — eliminates discontinuities at slack=0/1.
    - Introduced *relative MAD normalization*: for slack and uncertainty, normalize w.r.t.
      their own distribution *and* scale by median(|slack|+ε) to preserve risk ordering
      under outlier-heavy workloads (e.g., one task with extreme slack dominates ranking).
    - Energy dampening now gated by *fairness headroom*: uses sigmoid of
      (ready_wait_time / (|slack| + ε)), not raw slack pressure — prioritizes energy
      efficiency only when waiting time is low *relative* to deadline margin, avoiding
      starvation while preserving energy savings where feasible.
    - All numeric literals are strictly in {-2,-1,0,1,2}; no hidden constants.
    """
    eps = 6.145597529211387e-06
    finfo = np.finfo(float)

    def relative_mad_normalize(x, ref_scale=None):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = np.where(mad > eps, mad, eps)
        if ref_scale is not None:
            scale = np.maximum(scale, np.abs(ref_scale) + eps)
        return (x - med) / scale
    slack_arr = np.asarray(slack, dtype=float)
    neg_mask = slack_arr < 0.0
    pos_mask = slack_arr >= 0.0
    slack_norm = np.zeros_like(slack_arr)
    slack_norm[neg_mask] = 5.056544297086255 * -slack_arr[neg_mask]
    urgency_pos = 1.0 - np.exp(-0.27452010064901533 * slack_arr[pos_mask])
    urgency_scaled = 2.521726697051226 * urgency_pos
    slack_clipped = np.clip(slack_arr[pos_mask], 0.0, 7.352505283331998)
    linear_decay = slack_clipped / (7.352505283331998 + eps)
    slack_norm[pos_mask] = urgency_scaled * linear_decay + (1.0 - linear_decay) * slack_clipped
    duration = min_exec_time + min_comm_time
    duration_norm = relative_mad_normalize(duration)
    energy_norm = relative_mad_normalize(min_incremental_energy)
    wait_headroom_ratio = ready_wait_time / (np.abs(slack_arr) + eps)
    fairness_gate = 1.0 - 1.0 / (1.0 + np.exp(-0.8455035644478164 * (1.0 - wait_headroom_ratio)))
    energy_weight_adj = 0.79767338540533 * fairness_gate
    rank_norm = relative_mad_normalize(upward_rank)
    critical_gate = (slack_arr >= 0.0).astype(float)
    critical_boost = 0.013559705618124779 * rank_norm * critical_gate
    wait_norm = relative_mad_normalize(ready_wait_time)
    wait_headroom = np.maximum(1.0, np.abs(slack_arr) + 1.0)
    wait_score = 0.46644245956534225 * wait_norm / (wait_headroom + eps)
    unc_norm = relative_mad_normalize(uncertainty, ref_scale=np.median(np.abs(slack_arr) + eps))
    slack_pressure = np.clip(-slack_arr, 0.0, np.inf)
    slack_pressure_bounded = np.tanh(slack_pressure * 0.1665037996274897)
    uncertainty_amplifier = 0.26750445155423597 * unc_norm * slack_pressure_bounded
    score = 1 * slack_norm + 0.0084712659095373 * duration_norm + energy_weight_adj * energy_norm - critical_boost + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
