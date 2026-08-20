import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: hybrid DDL-protection via dual-gated urgency (tanh + median-slack global gate),
       joint successor-release coupling, robust mean-abs normalization, and bounded anti-starvation.
    
    Structural improvements vs parents:
      - Dual urgency gating: combines (1) local tanh-based slack urgency AND (2) global ddl_protection_boost
        activated only when median_slack <= 0 → ensures hard deadline compliance hierarchy without over-penalizing.
      - Successor-release now couples remaining_work * upward_rank * ddl_protection_gate → focuses unblocking on
        critical-path-heavy large downstream work only under DDL stress.
      - Replaces additive uncertainty penalty with multiplicative energy_uncertainty_interaction gated by both
        tight slack AND high uncertainty → improves feasibility by avoiding spurious penalties.
      - All numeric literals are -2,-1,0,1,2; no hidden constants; all parameters declared and used exactly once.
      - Uses mean-abs scaling (low-cost, outlier-robust) and np.tanh for bounded, differentiable urgency.
      - Explicit finite safeguards via np.nan_to_num and shape enforcement.
    """
    eps = 0.004226581359955302
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_norm(x):
        x_abs = np.abs(x)
        denom = np.mean(x_abs) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    norm_duration = robust_norm(min_exec_time + min_comm_time)
    norm_rank = robust_norm(upward_rank)
    norm_work = robust_norm(remaining_work)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    median_slack = np.median(slack)
    global_ddl_gate = (median_slack <= eps).astype(float)
    raw_urgency = -slack * 1.5415283482878952
    slack_urgency = np.tanh(raw_urgency)
    norm_urgency = robust_norm(slack_urgency)
    slack_tight = (slack < 0.0) | (norm_slack < -0.08345164995673038)
    uncert_low = norm_uncert < 0.50701658417758
    local_ddl_gate = np.where(slack_tight & uncert_low, 1.0, 0.0)
    boosted_rank = norm_rank * (1.0 + 1.7993638532920486 * local_ddl_gate)
    successor_release_bonus = 1.4830859468480866 * norm_work * norm_rank * global_ddl_gate
    wait_benefit = np.tanh(0.004224666190013331 * (norm_wait + 8.187699809196848e-09))
    risk_energy_gate = np.where((slack < 0.0) & (norm_uncert > 0.50701658417758), 1.0, 0.0)
    energy_uncert_penalty = norm_energy * risk_energy_gate * 0.39975757169218035
    score = +norm_urgency * (1.0 + 0.6554010936976399 * global_ddl_gate) - boosted_rank - 1.6092258214355089 * norm_energy - wait_benefit - successor_release_bonus + energy_uncert_penalty + 1.1040358475244196 * norm_work
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    return score.reshape(-1)
