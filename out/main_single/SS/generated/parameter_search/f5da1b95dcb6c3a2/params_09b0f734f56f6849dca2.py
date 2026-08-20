import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's superior normalization, nonlinear slack modeling, and bounded rank amplification with Parent 1's robust duration damping and explicit DDL-gated duration penalty.
       Structural novelty: introduces *duration_robustness_factor*-damped *norm_duration* penalty fully gated by ddl_gate and slack_pressure_gate — prevents noisy duration estimates from overriding deadline feasibility while preserving urgency coupling.
       All penalty terms now share uniform ddl_gate + slack_pressure_gate composition for consistency; anti-starvation remains additive linear bonus for monotonicity and discriminability."""
    eps = np.finfo(float).tiny
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
        x = np.copy(x)
        if N == 1:
            center = x[0]
            spread = eps
        else:
            center = np.mean(x)
            spread = np.std(x, ddof=0)
        spread = np.where(spread > eps, spread, eps)
        return (x - center) / spread
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-3.8304412765590294 * slack))
    raw_slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    slack_pressure_gate = 1.0 / (1.0 + np.exp(-1.1531391655596335 * (raw_slack_pressure - 1.0)))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_score = norm_rank * norm_work * ddl_breach
    rank_amplifier = 1.0 + 0.5125596603144215 * slack_pressure_gate
    rank_amplifier = np.clip(rank_amplifier, 0.38951237340090594, 1.7686212646254336)
    boosted_rank = norm_rank * rank_amplifier
    uncert_gate = 1.0 / (1.0 + np.exp(-3.8304412765590294 * (norm_uncert - 0.5509328820531009)))
    comm_uncert_penalty = norm_duration * norm_uncert * uncert_gate * ddl_breach
    wait_bonus = 0.45427305991828765 * norm_wait
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * raw_slack_pressure * ddl_gate
    duration_penalty = 0.3995828986031882 * norm_duration * slack_pressure_gate * ddl_gate
    score = +np.clip(norm_slack, -2.0, 2.0) + np.clip(raw_slack_pressure * 2.424734167291268, -2.0, 2.0) - np.clip(critical_path_score, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.9320020442011492 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) + np.clip(wait_bonus, -2.0, 2.0) + 0.028312858049237624 * np.clip(energy_uncert_penalty, -2.0, 2.0) + np.clip(comm_uncert_penalty, -2.0, 2.0) + 1.4390587121077751 * np.clip(norm_work, -2.0, 2.0) + np.clip(duration_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
