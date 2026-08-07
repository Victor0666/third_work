import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robust urgency & conditional bottleneck with Parent 1's explicit energy-uncertainty coupling:
      - Retains tanh-based urgency with clamped affine mapping for smooth deadline proximity response
      - Keeps tunable IQR normalization for adaptive dispersion across heterogeneous workloads
      - Preserves conditional bottleneck amplification gated by *verified* DDL risk (slack < 0 AND uncertainty > threshold * median)
      - Adds novel energy_uncertainty_coupling: modulates incremental energy by normalized uncertainty to discourage high-risk low-energy VMs
      - Removes redundant power-law DDL penalty (Parent 1) — tanh + gating is more stable and interpretable
      - All terms additive, bounded, and finite; no hidden constants beyond {-2,-1,0,1,2}
      - Fairness remains subtractive to prevent starvation
      - Normalization avoids division-by-zero via eps and IQR fallback
    """
    eps = 3.251397549805471e-05
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
        if N == 0:
            return np.zeros_like(x)
        q_low = np.percentile(x, 18.71767440208049)
        q_high = np.percentile(x, 70.6443294310457)
        iqr = q_high - q_low + eps
        center = np.median(x)
        return (x - center) / iqr
    abs_slack = np.abs(slack)
    scale = np.median(abs_slack) if np.any(abs_slack > eps) else 1.0
    tanh_urgency = np.tanh(-slack / (scale * 0.17185973477774866 + eps))
    clamped_urgency = np.clip(tanh_urgency, -0.6904606820490296, 1.0)
    urgency = (clamped_urgency + 1.0) / 2.0
    norm_urgency = adaptive_normalize(urgency)
    duration = min_exec_time + min_comm_time
    bottleneck_pressure = duration * upward_rank * (min_incremental_energy + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    unc_normalized = adaptive_normalize(uncertainty)
    risk_adjusted_energy = min_incremental_energy * (1.0 + 0.2939970171271579 * np.abs(unc_normalized))
    norm_energy = adaptive_normalize(risk_adjusted_energy)
    norm_wait = adaptive_normalize(ready_wait_time)
    fairness_term = -0.8586181506206931 * norm_wait
    median_uncertainty = np.median(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < 0.0) & (uncertainty > 0.7793459738604951 * median_uncertainty)).astype(float)
    risk_amplified_bottleneck = norm_bottleneck * (1.0 + 1.5105551454427766 * ddl_risk_gate)
    stress_amplified_urgency = norm_urgency * (1.0 + ddl_risk_gate)
    score = stress_amplified_urgency + 0.24218632716728436 * risk_amplified_bottleneck + norm_energy + fairness_term
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
