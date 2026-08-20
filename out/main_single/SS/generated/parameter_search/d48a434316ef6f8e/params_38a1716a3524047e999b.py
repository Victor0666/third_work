import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: uses robust quantile-based normalization;
       introduces successor-release interaction: min_exec_time × clipped_upward_rank × (slack <= 0);
       replaces sigmoid DDL gate with hard binary gating for strict feasibility-first enforcement;
       adds sign-preserving MAD-normalized slack for sharper urgency near zero;
       applies joint congestion gating only under both high ready_wait_time AND high uncertainty;
       enforces energy/uncertainty penalties exclusively when slack >= 0 (feasible regime)."""
    eps = 1.3295356666692727e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_range_normalize(x):
        x = np.copy(x)
        if N == 1:
            q_low = x[0]
            q_high = x[0]
            rng = eps
        else:
            q_low = np.quantile(x, 0.35163910514022456)
            q_high = np.quantile(x, 0.8134533077193731)
            rng = q_high - q_low
        spread = rng if rng > eps else eps
        center = (q_low + q_high) / 2.0
        return (x - center) / spread
    norm_slack = robust_range_normalize(slack)
    norm_energy = robust_range_normalize(min_incremental_energy)
    norm_exec = robust_range_normalize(min_exec_time)
    norm_comm = robust_range_normalize(min_comm_time)
    norm_rank = robust_range_normalize(upward_rank)
    norm_work = robust_range_normalize(remaining_work)
    norm_wait = robust_range_normalize(ready_wait_time)
    norm_uncert = robust_range_normalize(uncertainty)
    ddl_feasible = (slack >= 0.0).astype(float)
    if N == 1:
        med_slack = slack[0]
        mad_slack = eps
    else:
        med_slack = np.median(slack)
        mad_slack = np.median(np.abs(slack - med_slack))
    spread_slack = mad_slack if mad_slack > eps else eps
    signed_slack_norm = (slack - med_slack) / spread_slack
    ddl_breach = (slack <= 0.0).astype(float)
    critical_leverage = norm_rank * norm_work * ddl_breach
    clipped_rank = np.clip(norm_rank, 0.0, 2.0)
    successor_release = norm_exec * clipped_rank * ddl_breach
    wait_high = (norm_wait > 0.06982908178742718).astype(float)
    uncert_high = (norm_uncert > 0.06982908178742718).astype(float)
    congestion_gate = wait_high * uncert_high
    wait_benefit = 1.0 - np.exp(-0.2914179289767278 * (ready_wait_time + 0.00020426403420713981))
    wait_benefit = wait_benefit * (1.0 - congestion_gate)
    energy_uncert_penalty = norm_energy * norm_uncert * ddl_feasible
    slack_tightness = np.clip(1.0 - signed_slack_norm, 0.0, 2.0)
    energy_slack_penalty = norm_energy * slack_tightness * ddl_feasible
    score = +np.clip(signed_slack_norm, -2.0, 2.0) - np.clip(critical_leverage, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - 0.11842356377168536 * np.clip(norm_energy * ddl_feasible, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.6053722207321066 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.7453099713320621 * np.clip(energy_slack_penalty, -2.0, 2.0) + 0.5157634063062959 * np.clip(norm_work, -2.0, 2.0) + 1.8770217957626927 * np.clip(norm_rank * ddl_breach, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
