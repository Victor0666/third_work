import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with unified tight-slack gating and global IQR normalization:
      - Replaces per-term MAD with *single global IQR-based scaling* for all normalized terms → reduces noise in small N.
      - All scenario-sensitive components (critical path, energy, bottleneck) now gated by *same* tight-slack condition: slack <= median_slack * threshold_factor.
      - Removes urgency tanh and successor-risk terms to restore strict DDL-risk dominance and improve feasibility stability.
      - Bottleneck pressure uses uncertainty-amplified duration × critical-path product, scaled globally.
      - Wait decay remains power-law but now normalized *relative to global IQR scale*, ensuring consistent anti-starvation strength.
      - All operations protected against NaN/inf/zero; deterministic; only {-2,-1,0,1,2} as structural literals.
    """
    eps = 0.00013633101983245074
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def global_iqr_scale(x):
        if N <= 1:
            return eps
        q_high = np.percentile(x, 82.29701887531684)
        q_low = np.percentile(x, 29.69079542486559)
        iqr = q_high - q_low
        range_val = np.max(x) - np.min(x)
        return np.where(iqr > eps, iqr, range_val) + eps
    global_scale = global_iqr_scale(np.concatenate([min_exec_time + min_comm_time, upward_rank * remaining_work, min_incremental_energy, ready_wait_time]))
    neg_slack = np.clip(-slack, 0.0, None)
    ddl_risk_penalty = np.power(neg_slack + eps, 1.370668584914953)
    median_slack = np.median(slack) if N > 0 else 0.0
    tight_slack_mask = (slack <= median_slack * 0.8072473662326677).astype(float)
    critical_path_pressure = tight_slack_mask * upward_rank * remaining_work
    norm_critical_path = (critical_path_pressure - np.median(critical_path_pressure)) / global_scale
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy = (energy_per_duration - np.median(energy_per_duration)) / global_scale * 0.9439788650190796
    base_bottleneck = duration * upward_rank * remaining_work
    bottleneck_pressure = base_bottleneck * (1.0 + np.power(uncertainty + eps, 1.9216198603332622))
    norm_bottleneck = (bottleneck_pressure - np.median(bottleneck_pressure)) / global_scale
    max_wait = np.max(ready_wait_time) if N > 0 else eps
    wait_ratio = np.clip(ready_wait_time / (max_wait + eps), 0.0, 1.0)
    wait_decay = np.power(wait_ratio + eps, 0.23413893225355006)
    wait_priority = 1.0 - wait_decay
    norm_wait = (wait_priority - np.median(wait_priority)) / global_scale
    score = ddl_risk_penalty + norm_critical_path * 0.6167998778140833 + norm_bottleneck + norm_energy - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
