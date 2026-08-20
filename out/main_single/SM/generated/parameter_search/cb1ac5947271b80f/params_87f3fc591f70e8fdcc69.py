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
    eps = 2.6332281034610862e-06
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    duration = exec_t + comm_t
    feasibility_weight = 1.0 / (1.0 + np.exp(-3.7763998589701036 * slk))
    joint_features = np.abs(np.concatenate([duration + eps, energy + eps, rank + eps, rank * (work + eps) + eps]))
    q90 = np.percentile(joint_features, 84.35295036666813)
    rng_safe = np.where(np.isfinite(q90) & (q90 > eps), q90, eps)

    def joint_normalize(x):
        x = np.asarray(x)
        center = np.median(joint_features)
        return (x - center) / (0.6938493490462538 * rng_safe + eps)
    slack_norm = joint_normalize(slk)
    slack_penalty = np.where(slk < 0, 6.283963716216646 * slack_norm ** 2, -6.283963716216646 * 0.015503343069764159 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.2538931699843652 * feasibility_weight * joint_normalize(inv_energy + eps)
    rank_score = -1.24704155959699 * feasibility_weight * joint_normalize(rank + eps)
    dur_energy_prod = (duration + eps) * (energy + eps)
    dur_energy_urgency = np.power(dur_energy_prod, 1.230811352986222)
    dur_energy_score = feasibility_weight * joint_normalize(dur_energy_urgency + eps)
    wait_sat = 1.0 - np.exp(-0.21773337281124194 * wait)
    wait_score = -joint_normalize(wait_sat + eps)
    weighted_successor_load = rank * (work + eps)
    successor_release_score = -0.5409541013625943 * feasibility_weight * joint_normalize(weighted_successor_load + eps)
    score = slack_penalty + energy_score + rank_score + dur_energy_score + wait_score + successor_release_score
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
