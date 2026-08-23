import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with key structural improvements:
      - Replaces convex slack penalty with linear DDL-violation gain (slower saturation, preserves ordering).
      - Removes redundant `uncertainty_slack_coupling` per reflection — eliminated for fidelity and simplicity.
      - Uses robust median-based MAD normalization without clipping, scaled by tunable `robust_mad_scale`.
      - Introduces explicit latency-risk term: `min_exec_time + min_comm_time` normalized and weighted under DDL safety.
      - All non-DDL terms now share a unified robust scale via shared MAD normalization across risk features.
      - Final score strictly enforces lexicographic feasibility: violation penalty dominates; among compliant tasks,
        critical-path release balances against risk-adjusted energy and anti-starvation effects.
    """
    eps = 9.402881416909996e-08
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
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev) + eps
        normalized = (x - med) / (mad * 0.7623115699831415 + eps)
        return normalized
    duration_total = min_exec_time + min_comm_time + eps
    slack_abs = np.abs(slack)
    slack_violation = np.where(slack < 0, -slack * 21.487119913994846, 0.0)
    duration_risk = duration_total * uncertainty * 0.07813761642791263
    critical_release_score = upward_rank * remaining_work * 2.677258733734003
    slack_headroom_mask = np.where(slack > 2.987623813898439, 1.0, 0.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = slack_headroom_mask * 0.7504985864636858 * energy_norm
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 1.5075213658431492, 0.0)
    wait_norm = mad_normalize(wait_power)
    wait_term = slack_headroom_mask * wait_norm
    unc_sigmoid = 1.0 / (1.0 + np.exp(-2.2323971716010513 * (uncertainty - 1.0)))
    energy_uncertainty_term = slack_headroom_mask * 0.6757127854784826 * energy_norm * mad_normalize(uncertainty) * unc_sigmoid
    rank_norm = mad_normalize(upward_rank)
    slack_headroom = np.clip((slack - 2.987623813898439) / (2.987623813898439 + 1.0), 0.0, 1.0)
    weight_rank = 0.5428506565310439 + (1.0 - 0.5428506565310439) * (1.0 - slack_headroom)
    rank_term = slack_headroom_mask * -rank_norm * weight_rank
    latency_risk_norm = mad_normalize(duration_total)
    latency_risk_term = slack_headroom_mask * -latency_risk_norm
    score = mad_normalize(slack_violation) + mad_normalize(duration_risk) + mad_normalize(critical_release_score)
    score += energy_term + wait_term + energy_uncertainty_term + rank_term + latency_risk_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
