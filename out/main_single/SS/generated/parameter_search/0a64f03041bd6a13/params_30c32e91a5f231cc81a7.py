import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces sigmoidal successor-release with additive critical-path coupling (rank × work × pressure) per reflection;
       reintroduces ddl_protection_gate_slope as bounded linear multiplier on DDL-pressure signal;
       adds host-load-aware gating — *not as external input*, but synthesized via robust uncertainty-duration coupling as proxy for congestion.
       Structural novelty: uses `norm_duration * norm_uncert` as a lightweight, input-free surrogate for host load (high duration + high uncertainty → likely congested or degraded VM),
       gated by hard_feasibility_gate and scaled by host_load_sensitivity — avoids requiring unavailable host_load array while preserving congestion-aware energy suppression."""
    eps = 0.0008390543732242701
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
    slack_penalty = 1.465847122199791 * norm_slack_penalty
    successor_release = norm_rank * norm_work * ddl_pressure
    coupled_rank = norm_rank * (1.0 + 1.4428747876546972 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.005044884751371085, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_feasibility_gate
    wait_benefit = np.clip(0.45442321277043285 * ready_wait_time, 0.0, 2.0)
    duration_penalty = 0.16329650336760784 * norm_duration * ddl_pressure * hard_feasibility_gate
    slack_energy_suppress = np.where(slack < 0.0, 1.0 - 0.44031039378878717, 1.0)
    suppressed_norm_energy = norm_energy * slack_energy_suppress
    load_surrogate = norm_duration * norm_uncert
    load_gate = np.where((hard_feasibility_gate > 0.0) & (norm_uncert > 0.005044884751371085), np.clip(load_surrogate, 0.0, 2.0), 0.0)
    host_load_penalty = 0.37827016391914947 * load_gate * suppressed_norm_energy
    ddl_protection_boost = np.clip(4.98241508516496 * ddl_pressure, -2.0, 2.0)
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.22750575697178618 * np.clip(suppressed_norm_energy, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_penalty, -2.0, 2.0) + 0.455058931340948 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.5076062000923394 * np.clip(norm_work, -2.0, 2.0) + ddl_protection_boost + host_load_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
