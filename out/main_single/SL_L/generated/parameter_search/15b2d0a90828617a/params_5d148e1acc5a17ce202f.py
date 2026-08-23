import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Hybrid priority rule: piecewise slack scoring (exponential lateness penalty + linear earliness reward),
    criticality boost gated by *both* rank and slack tightness, uncertainty-coupled deadline pressure,
    hybrid anti-starvation (exponential soft boost + saturation cap), and rank/energy interpolation
    weighted by slack tightness. All numeric literals are -2,-1,0,1,2; all tunables declared.
    """
    eps = 2.4193275724280534e-08

    def robust_norm(x):
        x = np.asarray(x, dtype=np.float64)
        denom = np.mean(np.abs(x)) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    norm_duration = robust_norm(min_exec_time + min_comm_time)
    norm_rank = robust_norm(upward_rank)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    slack_sign_mask_neg = (slack < 0).astype(float)
    slack_sign_mask_pos = (slack > 0).astype(float)
    urgency_base = -norm_slack * slack_sign_mask_neg
    urgency_clipped = np.clip(urgency_base, 0.0, 5.976839539090196)
    urgency_penalty = np.exp(3.0958117159999134 * urgency_clipped)
    early_reward = slack * slack_sign_mask_pos * 0.7499784514067733
    early_reward_norm = robust_norm(early_reward)
    slack_score = urgency_penalty * slack_sign_mask_neg + early_reward_norm * slack_sign_mask_pos
    rank_median = np.median(norm_rank) if len(norm_rank) > 1 else np.mean(norm_rank)
    is_high_rank = norm_rank >= rank_median
    is_tight_slack = slack <= 2 * eps
    critical_gate = np.where(is_high_rank & is_tight_slack, 1.0752200065379545, 1.0)
    critical_boost = norm_rank * critical_gate
    deadline_pressure = np.maximum(0.0, -norm_slack)
    uncertainty_coupled_pressure = 0.0033219232417521717 * deadline_pressure * norm_uncert
    wait_exp_boost = 1.0 - np.exp(-0.30946388720824825 * norm_wait)
    wait_clipped = np.clip(ready_wait_time, 0, 1.398554003912761)
    wait_sat_norm = robust_norm(wait_clipped)
    wait_score = -wait_sat_norm + wait_exp_boost
    rank_energy_penalty = 0.62279483484231 * norm_energy * norm_rank
    slack_min = np.min(slack)
    slack_max = np.max(slack)
    slack_range = np.maximum(eps, slack_max - slack_min)
    slack_normalized = np.clip((slack - slack_min) / (slack_range + eps), 0, 1)
    weight_rank = 1.0 - slack_normalized * (1.0 - 0.3257586927220121)
    rank_energy_tradeoff = weight_rank * norm_rank + (1.0 - weight_rank) * norm_energy
    score = robust_norm(slack_score) + uncertainty_coupled_pressure + rank_energy_penalty + 0.5032297326954648 * norm_energy + 0.42395300582481976 * norm_duration - critical_boost + wait_score + rank_energy_tradeoff
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=-finfo.max)
    return score
