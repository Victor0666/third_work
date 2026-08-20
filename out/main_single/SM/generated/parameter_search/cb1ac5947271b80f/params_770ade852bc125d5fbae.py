import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with smooth DDL feasibility gating, unified joint normalization,
    and eliminated percentile parameter redundancy.
    
    Key structural improvements:
      - Replaced hard `slk >= 0` gate with smooth logistic feasibility weight: 
        `sigma(slack * slack_feasibility_weight)` — avoids discontinuities near slack=0.
      - Unified joint normalization over duration, energy, rank, and successor-load using single
        robust 90th-quantile scale — improves stability and reduces parameter coupling.
      - Removed `pctl_low`/`pctl_high`; fixed lower bound implied by median centering.
      - All components share identical normalization context; `positive_slack_urgency_gain` now tunable.
    """
    eps = 2.613758901181777e-07
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    duration = exec_t + comm_t
    feasibility_weight = 1.0 / (1.0 + np.exp(-2.626436838164104 * slk))
    joint_features = np.abs(np.concatenate([duration + eps, energy + eps, rank + eps, rank * (work + eps) + eps]))
    q90 = np.percentile(joint_features, 80.7755856205532)
    rng_safe = np.where(np.isfinite(q90) & (q90 > eps), q90, eps)

    def joint_normalize(x):
        x = np.asarray(x)
        center = np.median(joint_features)
        return (x - center) / (1.3012070541438236 * rng_safe + eps)
    slack_norm = joint_normalize(slk)
    slack_penalty = np.where(slk < 0, 1.8725665244746557 * slack_norm ** 2, -1.8725665244746557 * 0.7590652593015289 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.0629885878843535 * feasibility_weight * joint_normalize(inv_energy + eps)
    rank_score = -2.2715724632724212 * feasibility_weight * joint_normalize(rank + eps)
    dur_energy_prod = (duration + eps) * (energy + eps)
    dur_energy_urgency = np.power(dur_energy_prod, 1.4929473554574995)
    dur_energy_score = feasibility_weight * joint_normalize(dur_energy_urgency + eps)
    wait_sat = 1.0 - np.exp(-0.22751518607683438 * wait)
    wait_score = -joint_normalize(wait_sat + eps)
    weighted_successor_load = rank * (work + eps)
    successor_release_score = -0.025582913561895317 * feasibility_weight * joint_normalize(weighted_successor_load + eps)
    score = slack_penalty + energy_score + rank_score + dur_energy_score + wait_score + successor_release_score
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
