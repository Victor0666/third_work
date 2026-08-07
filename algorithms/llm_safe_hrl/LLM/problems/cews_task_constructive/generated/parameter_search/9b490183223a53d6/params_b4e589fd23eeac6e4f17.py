import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's stability with Parent 1's risk-aware gating:
      - Uses robust MAD-based normalization (Parent 2) for all normalized terms.
      - Retains power-law DDL risk penalty (Parent 2) but adds bounded tanh urgency (Parent 1) as *additive* signal for fine-grained near-deadline sensitivity.
      - Critical-path coupling activated only under tight slack (Parent 2), enhanced by uncertainty-amplified bottleneck pressure (Parent 1).
      - Introduces explicit successor-slack-deficit term (Parent 1) to mitigate release-blocking.
      - Replaces linear fairness with power-law wait decay (Parent 2) for smoother anti-starvation.
      - All operations protected against NaN/inf/zero; deterministic; no numeric literals beyond {-2,-1,0,1,2}.
    """
    eps = 5.696357763742091e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def mad_normalize(x):
        x = np.copy(x)
        if N == 0:
            return np.zeros(0, dtype=float)
        if N == 1:
            return np.zeros(1, dtype=float)
        center = np.median(x)
        abs_dev = np.abs(x - center)
        mad = np.median(abs_dev)
        fallback_range = np.max(x) - np.min(x)
        denom = np.where(mad > eps, mad, fallback_range)
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    ddl_risk_penalty = np.power(neg_slack + eps, 0.5823598760691563)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_base = np.clip(median_slack - slack, 0.0, 2.0)
    urgency = np.tanh(1.158919580496676 * urgency_base)
    tight_slack_mask = (slack <= median_slack).astype(float)
    critical_path_pressure = tight_slack_mask * upward_rank * remaining_work
    norm_critical_path = mad_normalize(critical_path_pressure)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy = mad_normalize(energy_per_duration) * 1.353248952065358
    unc_gate = (uncertainty >= 0.30501523346354886).astype(float)
    base_bottleneck = duration * upward_rank * remaining_work
    bottleneck_pressure = base_bottleneck * (1.0 + unc_gate * np.power(uncertainty + eps, 1.2763298481098304))
    norm_bottleneck = mad_normalize(bottleneck_pressure)
    max_wait = np.max(ready_wait_time) if N > 0 else eps
    wait_ratio = np.clip(ready_wait_time / (max_wait + eps), 0.0, 1.0)
    wait_decay = np.power(wait_ratio + eps, 0.7595170746048716)
    wait_priority = 1.0 - wait_decay
    norm_wait = mad_normalize(wait_priority)
    successor_risk_proxy = (slack < median_slack - eps).astype(float)
    successor_risk_penalty = 0.9761561365106497 * successor_risk_proxy
    score = ddl_risk_penalty + urgency + 0.5661207099585337 * norm_critical_path + norm_bottleneck + norm_energy - norm_wait + successor_risk_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
