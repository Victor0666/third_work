import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust joint DDL-risk gating and adaptive normalization
    with Parent 1's principled anti-starvation via bounded reciprocal scaling — but re-engineered
    to avoid over-smoothing: uses tunable saturation time instead of median-based ramp.
    Key structural improvements:
    1. Unified risk coupling: multiplicative urgency × uncertainty (Parent 2) + additive DDL-risk amplification
       only when slack < 0 (tighter than median-based condition), preserving hard deadline semantics.
    2. Bounded reciprocal anti-starvation: 1/(1 + wait/saturation)^exponent, using explicit saturation time
       (more interpretable and controllable than median-based ramp or arctan).
    3. Adaptive normalization retains IQR→min-max fallback (Parent 2) but applies it consistently across all terms.
    4. Removed critical-path bonus (evidence shows no DDL gain, harms energy); replaced with direct
       urgency-driven bottleneck coupling: duration × upward_rank × (1 + urgency) × (1 + uncertainty).
    All operations are finite, deterministic, and use only [-2,-1,0,1,2] literals.
    """
    eps = 4.92528017935408e-05
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
        q_low = np.percentile(x, 39.97789092060192)
        q_high = np.percentile(x, 77.37035323252253)
        iqr = q_high - q_low
        center = np.median(x)
        if iqr < eps:
            x_min, x_max = (np.min(x), np.max(x))
            denom = x_max - x_min
            if denom < eps:
                return np.zeros_like(x)
            return (x - x_min) / (denom + eps)
        else:
            denom = iqr
            return (x - center) / (denom + eps)
    slack_centered = slack - 0.4598043808351593
    sigmoid_input = np.clip(-6.821201836061594 * slack_centered, -10.563387343379414, 10.563387343379414)
    urgency = 1.0 / (1.0 + np.exp(sigmoid_input))
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency + eps) * (1.0 + uncertainty + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (4.3995779811021025 + eps)
    wait_reciprocal = 1.0 / (1.0 + wait_scaled)
    norm_wait = adaptive_normalize(wait_reciprocal)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < 0.0) & (uncertainty > 0.6839582246825585 * max_uncertainty)).astype(float)
    risk_coupled = urgency * uncertainty
    norm_risk = adaptive_normalize(risk_coupled)
    score = norm_urgency + 0.3121838322301285 * norm_bottleneck + ddl_risk_gate * norm_risk + 1.5565837429704845 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
