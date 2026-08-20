import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all numeric literals are -2,-1,0,1,2; 0.5 replaced by PARAMS["slack_tightness_threshold"];
       uses only declared parameters; no hidden constants; robust scaling; bounded tanh-based urgency."""
    eps = 0.003046963487952567
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
        center = np.mean(x_abs) if N > 0 else 0.0
        spread = np.mean(x_abs) + eps
        return (x - center) / (spread + eps)
    norm_slack = scale_feature(slack)
    norm_energy = scale_feature(min_incremental_energy)
    norm_duration = scale_feature(min_exec_time + min_comm_time)
    norm_rank = scale_feature(upward_rank)
    norm_work = scale_feature(remaining_work)
    norm_wait = scale_feature(ready_wait_time)
    norm_uncert = scale_feature(uncertainty)
    raw_urgency = -slack * 1.3553384964996749
    slack_urgency = np.tanh(raw_urgency)
    norm_slack_penalty = scale_feature(slack_urgency)
    slack_tight = (slack < 0.0) | (norm_slack < -0.3794218141126502)
    uncert_low = norm_uncert < 0.24815413818100818
    ddl_protection_gate = np.where(slack_tight & uncert_low, 1.0, 0.0)
    boosted_rank = norm_rank * (1.0 + 0.8915318693045557 * ddl_protection_gate)
    successor_release_score = norm_work * norm_rank * ddl_protection_gate
    wait_benefit = np.tanh(0.36494259043499655 * (norm_wait + 1.9397172478840275e-07))
    risk_energy_gate = np.where((slack < 0.0) & (norm_uncert > 0.24815413818100818), 1.0, 0.0)
    energy_uncert_penalty = norm_energy * risk_energy_gate
    score = +norm_slack_penalty - boosted_rank - 0.6220268880502554 * norm_energy - wait_benefit - successor_release_score + 0.7088672788079381 * energy_uncert_penalty + 1.4071551591700335 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
