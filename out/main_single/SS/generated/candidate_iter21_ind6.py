import numpy as np
RULE_METADATA = {'structure_hash': 'ef6329dc431abf3e75ab3673924fbaf4dd74dfba2a676b72541e5c11b37c08d8', 'parameter_schema_hash': 'c553c59ea5141df47b7967b7aaf1cbe24f39b34de58098090f8e2ef9ce4e3f3c', 'best_parameter_hash': 'b036be2006051193ff1728888d361c39392bdf0c577f0f2e9f748df0796f6d97', 'best_parameters': {'epsilon': 0.0007313558403276415, 'slack_penalty_exponent': 1.9956115655463886, 'criticality_scale': 1.0843776607790792, 'energy_sensitivity': 0.6720879258249662, 'uncertainty_gate_threshold': 0.10149144358711698, 'slack_pressure_gate_steepness': 1.0016345698774574, 'remaining_work_weight': 1.4187535163314902, 'wait_decay': 0.36465296749167175, 'ddl_protection_gate_slope': 3.4667943574330846, 'energy_uncertainty_interaction': 0.6413886322012892}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'd19107e031f05a4e931e50b84fdf7cab1f0a7ad2f9e2a5a0aa54a792d713bb33', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces unstable power-law rank-slack coupling with *clipped linear hinge* for robust monotonic urgency;
       unifies all risk terms under a single *joint slack-uncertainty gate* to reduce over-parameterization;
       introduces *normalized wait-time saturation* using softplus instead of exp to prevent overflow;
       removes redundant duration_robustness — now fully absorbed into joint_gate and slack_pressure scaling;
       retains validated critical_path_leverage and DDL-protection semantics while tightening AST depth and eliminating inf-producing branches."""
    eps = 0.0007313558403276415
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def median_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    joint_gate = 1.0 / (1.0 + np.exp(-3.4667943574330846 * slack)) * 1.0 / (1.0 + np.exp(3.4667943574330846 * (norm_uncert - 0.10149144358711698)))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.9956115655463886
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_hinge = np.clip(-norm_slack, 0.0, 1.0)
    rank_slack_coupling = 1.0 + 1.0843776607790792 * slack_hinge
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.0016345698774574 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * rank_slack_coupling * (1.0 + 1.0843776607790792 * rank_gate)
    wait_benefit = np.log1p(np.exp(-0.36465296749167175 * (ready_wait_time + eps)))
    duration_risk_score = norm_duration * norm_uncert * joint_gate
    energy_uncert_penalty = norm_energy * norm_uncert * joint_gate
    energy_slack_penalty = norm_energy * slack_pressure * joint_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.6720879258249662 * np.clip(norm_energy * joint_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.6413886322012892 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.9956115655463886 * np.clip(energy_slack_penalty, -2.0, 2.0) + 1.4187535163314902 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
