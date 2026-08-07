import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's robust DDL-risk gate and fairness with Parent 1's additive uncertainty coupling and urgency clamping.
    
    Key structural improvements:
      - Retains Parent 2's clean separation: urgency (deadline safety) vs bottleneck (critical-path + energy) vs fairness (anti-starvation)
      - Integrates Parent 1's *additive* uncertainty coupling into bottleneck_pressure to avoid explosion under high uncertainty
      - Uses Parent 1's `urgency_clamp_lower` on raw tanh output before affine scaling to [0,1], improving stability for marginally negative slack
      - Keeps unconditional fairness term (`-wait_fairness_weight * norm_wait`) for starvation prevention across all regimes
      - Joint DDL-risk gate remains `(slack < 0) & (uncertainty > median_uncertainty * threshold)` — simple, reliable, quantile-free
      - All operations are finite, deterministic, and numerically safe; no hidden constants beyond {-2,-1,0,1,2}
    """
    eps = 4.573511775057492e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def adaptive_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 24.88405483889348)
        q_high = np.percentile(x, 81.63967382841878)
        iqr = q_high - q_low + eps
        center = np.median(x)
        return (x - center) / iqr
    abs_slack = np.abs(slack)
    scale = np.median(abs_slack) if np.any(abs_slack > eps) else 1.0
    tanh_urgency = np.tanh(-slack / (scale * 1.4664534189091953 + eps))
    clamped_urgency = np.clip(tanh_urgency, -0.8624710667004252, 1.0)
    urgency = (clamped_urgency + 1.0) / 2.0
    norm_urgency = adaptive_normalize(urgency)
    duration = min_exec_time + min_comm_time
    bottleneck_pressure = duration * upward_rank * (min_incremental_energy + eps) + 0.30301339408339584 * uncertainty * duration
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_wait = adaptive_normalize(ready_wait_time)
    fairness_term = -0.010478587321044399 * norm_wait
    median_uncertainty = np.median(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < 0.0) & (uncertainty > 0.6842759338985814 * median_uncertainty)).astype(float)
    stress_amplified_urgency = norm_urgency * (1.0 + ddl_risk_gate)
    score = stress_amplified_urgency + 1.0775536322845938 * norm_bottleneck + fairness_term
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
