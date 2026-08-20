import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: reinstates HARD binary feasibility gating (not sigmoidal) per reflection;
       unifies all features under robust MAD normalization (including slack) for outlier resilience;
       promotes explicit successor-release coupling as additive term — now *unweighted* to avoid parameter bloat;
       removes redundant sigmoid-derived gates and preserves sharp deadline boundary behavior;
       introduces bounded linear wait_benefit modulation by (1 - ddl_pressure) *only* under feasibility."""
    eps = 0.0013211352915196545
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad + eps
        return (x - med) / spread
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    hard_feasibility_gate = np.where(slack < 0.0, 0.0, 1.0)
    ddl_pressure = np.clip(-norm_slack, 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_penalty = 0.7690591268445854 * norm_slack_penalty
    successor_release = norm_rank * norm_work * ddl_pressure
    coupled_rank = norm_rank * (1.0 + 1.760498823030472 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.501005487254324, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_feasibility_gate
    wait_benefit = np.clip(0.9994992062798019 * ready_wait_time * (1.0 - ddl_pressure) * hard_feasibility_gate, 0.0, 2.0)
    duration_penalty = 0.5867881952877331 * norm_duration * ddl_pressure * hard_feasibility_gate
    slack_energy_suppress = np.where(slack < 0.0, 1.0 - 0.7810345376144066, 1.0)
    suppressed_norm_energy = norm_energy * slack_energy_suppress
    load_surrogate = norm_duration * norm_uncert
    load_gate = np.where((hard_feasibility_gate > 0.0) & (norm_uncert > 0.501005487254324), np.clip(load_surrogate, 0.0, 2.0), 0.0)
    host_load_penalty = 0.9949862900369417 * load_gate * suppressed_norm_energy
    ddl_protection_boost = np.clip(5.830950497775655 * ddl_pressure, -2.0, 2.0)
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.5105484394663269 * np.clip(suppressed_norm_energy, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_penalty, -2.0, 2.0) + 0.16763321334672793 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.1414600588144617 * np.clip(norm_work, -2.0, 2.0) + ddl_protection_boost + host_load_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
