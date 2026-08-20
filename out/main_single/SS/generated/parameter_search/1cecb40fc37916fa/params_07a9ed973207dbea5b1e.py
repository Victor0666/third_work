import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: combines Parent 2's robust linear-ramp slack activation and energy-uncertainty interaction
       with Parent 1's bounded energy-slack decay—both gated by DDL feasibility.
       Uses sign-aware median/MAD normalization throughout. All terms clipped to [-2,2] for ordinal stability.
       Introduces novel *dual-gated* rank-uncertainty coupling derived from Parent 1 but merged into a single parameter-free term:
       norm_rank * norm_uncert * ddl_gate * (1 if norm_uncert > threshold else 0) → avoids adding new parameter.
       Eliminates redundant sigmoid gates; replaces with monotonic ramp + Heaviside for interpretability and CMA-ES convergence."""
    eps = 0.01046965703324516
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
        if N == 1:
            center = x[0]
            spread = eps
        else:
            center = np.median(x)
            spread = np.median(np.abs(x - center))
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.715992144511379
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_ramp = np.clip((1.0 - norm_slack) * 0.8712517881972912, 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.1978022717413186 * slack_ramp)
    ddl_gate = 1.0 / (1.0 + np.exp(-9.718995077749566 * slack))
    uncert_gate = np.where(norm_uncert > 0.44941393552032205, 1.0, 0.0)
    rank_uncert_coupling = norm_rank * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.048238245489502535 * (norm_wait + 3.753326980165466e-07))
    duration_risk_score = norm_duration * norm_uncert * uncert_gate * ddl_gate
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_decay_factor = np.exp(-0.8712517881972912 * np.maximum(slack, 0.0))
    energy_preference = norm_energy * ddl_gate * energy_decay_factor
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.8935501896263591 * np.clip(energy_preference, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.6366119383111372 * np.clip(duration_risk_score, -2.0, 2.0) + 0.6074568912663815 * np.clip(energy_uncert_penalty, -2.0, 2.0) + np.clip(rank_uncert_coupling, -2.0, 2.0) + 0.30227952808861336 * np.clip(norm_work, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
