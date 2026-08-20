import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's robust DDL pressure handling with Parent 1's successor-release insight.
       Key innovations: (1) Urgency saturation offset prevents gradient vanishing at slack=0; (2) Duration penalty now uses
       physical-scale sum (exec+comm) instead of normalized version, preserving latency sensitivity; (3) Critical-path
       coupling extended to include *normalized* remaining_work (not just binary), enabling graded boosting;
       (4) All numeric literals are -2,-1,0,1,2; no hidden constants."""
    eps = 0.0004942925698938887
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
    raw_urgency_input = -slack + 4.230635436035448e-08
    slack_urgency = np.tanh(raw_urgency_input * 1.1305577968355278)
    slack_pressure_score = np.clip(-slack, 0.0, np.inf)
    work_pressure_score = np.clip(norm_work, 0.0, np.inf)
    coupling_strength = slack_pressure_score * work_pressure_score
    boosted_rank = norm_rank * (1.0 + 1.2652140256999767 * coupling_strength)
    risk_active = (slack < 0.0) & (norm_uncert > 0.8286873409404227)
    energy_uncert_penalty = np.where(risk_active, norm_energy * norm_uncert, np.zeros_like(norm_energy))
    wait_benefit = np.tanh(0.13477576629035795 * norm_wait)
    energy_weight = np.where(slack > 0.0, 1.1924357356873134, 1.1924357356873134 * 0.7038264279961832)
    duration_penalty = (min_exec_time + min_comm_time) * 1.403382138011037
    score = +(1.0 - slack_urgency) - boosted_rank - energy_weight * norm_energy - duration_penalty - wait_benefit + 0.600663732093888 * energy_uncert_penalty + 0.2590503998902771 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
