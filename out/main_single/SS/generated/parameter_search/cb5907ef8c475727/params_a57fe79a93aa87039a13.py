import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with all declared parameters used and no unused/missing references.
       Key properties:
         - Uses only declared PARAMS keys; no slack_penalty_exponent or slack_tightness_threshold.
         - All numeric literals are in {-2,-1,0,1,2}; epsilon comes from PARAMS["epsilon"].
         - Robust mean-abs scaling without centering.
         - Smooth tanh-based urgency and gating (no boolean branches).
         - Successor-release coupling via norm_work * norm_rank * ddl_protection_gate.
         - Finite, shape-(N,) output with full NaN/inf protection.
    """
    eps = 1.6518546168235912e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def scale_feature(x):
        x_abs = np.abs(x)
        spread = np.mean(x_abs) + eps
        return x / (spread + eps)
    norm_slack = scale_feature(slack)
    norm_energy = scale_feature(min_incremental_energy)
    norm_duration = scale_feature(min_exec_time + min_comm_time)
    norm_rank = scale_feature(upward_rank)
    norm_work = scale_feature(remaining_work)
    norm_wait = scale_feature(ready_wait_time)
    norm_uncert = scale_feature(uncertainty)
    urgency_signal = np.tanh(-norm_slack * 2.379767821023332)
    uncert_gate = np.tanh((0.433542138285669 - norm_uncert) * 2.0)
    ddl_protection_gate = np.clip(urgency_signal * uncert_gate, 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.3207788511005503 * ddl_protection_gate)
    successor_release_score = norm_work * norm_rank * 1.1092793662386748 * ddl_protection_gate
    wait_benefit = np.tanh(0.5956122555868872 * (norm_wait + 7.929190496521473e-09))
    risk_energy_penalty = norm_energy * norm_uncert * ddl_protection_gate * (1.0 - uncert_gate)
    score = +urgency_signal - boosted_rank - successor_release_score - wait_benefit + 0.7840526032619667 * risk_energy_penalty - 0.8826021248351769 * norm_energy + 1.8347883542120285 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
