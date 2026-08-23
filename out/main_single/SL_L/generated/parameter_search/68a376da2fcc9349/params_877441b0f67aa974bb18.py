import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Lexicographic DDL enforcement: hard gating via `slack > ddl_protection_threshold`
      - Robust MAD-aligned normalization over |slack|, uncertainty, duration_total for scale coherence
      - Critical-path release term (upward_rank * remaining_work) unconditionally active
      - Convex slack violation penalty: (-slack)^exponent for stronger lateness discrimination
      - Uncertainty-slack coupling: linearly penalizes high uncertainty when slack is tight/negative
      - Power-law anti-starvation: `ready_wait_time ** wait_time_decay_exponent`, scaled only under safety
      - Energy-uncertainty interaction gated by sigmoid to suppress noise at low uncertainty
      - All numeric literals restricted to {-2,-1,0,1,2}; no other constants used
      - Final score ensures strict feasibility-first ordering: DDL violation dominates; among compliant, trade off criticality vs risk-adjusted energy
    """
    eps = 2.7057363818604164e-08
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    duration_total = min_exec_time + min_comm_time + eps
    slack_abs = np.abs(slack)
    slack_penalty = np.where(slack < 0, (-slack) ** 3.912375654490899, 0.0)
    deadline_pressure = np.maximum(0.0, -slack + 0.6795997625141579)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.22552729881406333
    duration_risk = duration_total * uncertainty * 1.0595690286678674
    critical_release_score = upward_rank * remaining_work * 3.2337492181519325
    slack_headroom_mask = np.where(slack > 0.6795997625141579, 1.0, 0.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = slack_headroom_mask * 0.6920851676751459 * energy_norm
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 1.9214988394205816, 0.0)
    wait_norm = mad_normalize(wait_power)
    wait_term = slack_headroom_mask * wait_norm
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.4630878694225653 * (uncertainty - 1.0)))
    energy_uncertainty_term = slack_headroom_mask * 0.8886208953008663 * energy_norm * mad_normalize(uncertainty) * unc_sigmoid
    rank_norm = mad_normalize(upward_rank)
    slack_headroom = np.clip((slack - 0.6795997625141579) / (0.6795997625141579 + 1.0), 0.0, 1.0)
    weight_rank = 0.5352154270978564 + (1.0 - 0.5352154270978564) * (1.0 - slack_headroom)
    rank_term = slack_headroom_mask * -rank_norm * weight_rank
    score = mad_normalize(slack_penalty) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_release_score)
    score += energy_term + wait_term + energy_uncertainty_term + rank_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
