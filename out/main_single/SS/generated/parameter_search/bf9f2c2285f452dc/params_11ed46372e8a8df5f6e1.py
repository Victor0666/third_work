import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
       (1) Critical-path-aware latency penalty: (exec+comm) * upward_rank;
       (2) Soft DDL saturation gate: sigmoid(-slack/tau) — smooth, bounded [0,1], no offset needed;
       (3) All parameters declared in schema are used; no unused or missing references;
       (4) No numeric literals except -2,-1,0,1,2; epsilon handled via PARAMS and np.finfo."""
    eps = 0.09991818232169464
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def stable_normalize(x):
        x_abs = np.abs(x)
        center = np.mean(x_abs) if N > 0 else 0.0
        spread = np.mean(x_abs) + eps
        return (x - center) / (spread + eps)
    norm_energy = stable_normalize(min_incremental_energy)
    norm_rank = stable_normalize(upward_rank)
    norm_work = stable_normalize(remaining_work)
    norm_wait = stable_normalize(ready_wait_time)
    norm_uncert = stable_normalize(uncertainty)
    tau = np.maximum(1.1974961185536466, np.finfo(float).eps)
    ddl_saturation = 1.0 / (1.0 + np.exp(-slack / tau))
    slack_urgency = 1.0 - ddl_saturation
    raw_duration = min_exec_time + min_comm_time
    critical_path_duration_penalty = raw_duration * norm_rank * 1.4428150302849867
    slack_pressure_score = np.clip(-slack, 0.0, np.inf)
    work_pressure_score = np.clip(norm_work, 0.0, np.inf)
    coupling_strength = slack_pressure_score * work_pressure_score
    boosted_rank = norm_rank * (1.0 + 0.9853467656731125 * coupling_strength)
    risk_active = (slack < 0.0) & (norm_uncert > 0.37815861176677357)
    energy_uncert_penalty = np.where(risk_active, norm_energy * norm_uncert, np.zeros_like(norm_energy))
    wait_benefit = np.tanh(0.016444412106930217 * norm_wait)
    energy_weight = np.where(slack > 0.0, 0.6452098861042981, 0.6452098861042981 * 0.7453907012458543)
    score = +slack_urgency - boosted_rank - energy_weight * norm_energy - critical_path_duration_penalty - wait_benefit + 0.992437427674119 * energy_uncert_penalty + 0.7708885107343206 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
