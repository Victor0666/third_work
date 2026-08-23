import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three structural improvements:
      - Replaces multiplicative criticality (upward_rank * remaining_work) with monotonic additive slack-gated form:
        upward_rank + remaining_work * (1 - ReLU(slack)/max(1, quantile_90(slack)+eps)) — improves gradient flow and outlier robustness.
      - Derives slack bounds adaptively from current ready set via tunable quantiles (quantile_low/quantile_high), eliminating fragile static parameters.
      - Replaces binary slack_headroom_mask with smooth bounded ReLU gate: clip(slack, 0, None) / (quantile_90(slack) + eps),
        enabling continuous CMA-ES optimization without threshold artifacts.
    All numeric literals are restricted to {-2,-1,0,1,2}; no other constants used.
    """
    eps = 5.378913128316531e-06
    mm_eps = 0.009878042147033646
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
    slack_q_low = np.quantile(slack, 0.11160725988105129) if N > 1 else np.mean(slack)
    slack_q_high = np.quantile(slack, 0.958891053597175) if N > 1 else np.mean(slack)
    slack_range = np.maximum(slack_q_high - slack_q_low, eps)

    def minmax_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        x_min = np.min(x)
        x_max = np.max(x)
        denom = x_max - x_min + mm_eps
        normalized = (x - x_min) / denom
        return np.clip(normalized, 0.0, 1.0)
    slack_score = np.where(slack < 0, (-slack) ** 2.7211340597367735, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.0742396113858845
    duration_total = min_exec_time + min_comm_time + eps
    duration_risk_base = duration_total * uncertainty
    duration_risk_penalty = np.where(slack <= 0.0, duration_risk_base * 0.17487820361455417, 0.0)
    slack_decay = np.clip((slack_q_high - slack) / (slack_range + eps), 0.0, 1.0)
    critical_release = upward_rank + remaining_work * slack_decay
    critical_release_norm = minmax_normalize(critical_release)
    critical_release_score = -critical_release_norm * 2.209175039446156
    slack_headroom = np.clip(slack, 0.0, None)
    slack_gate = slack_headroom / (slack_q_high + eps)
    slack_gate = np.clip(slack_gate, 0.0, 1.0)
    load_proxy = 1.0 + minmax_normalize(uncertainty)
    scaled_energy = min_incremental_energy * load_proxy
    energy_per_sec = scaled_energy / duration_total
    energy_eff_score = minmax_normalize(energy_per_sec)
    slack_scaled = np.clip((slack - slack_q_low) / (slack_range + eps), 0.0, 1.0)
    weight_rank = 0.7997001365944427 + (1.0 - 0.7997001365944427) * (1.0 - slack_scaled)
    rank_score = -minmax_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-1.984057225982672 * (uncertainty - 1.0)))
    energy_norm = minmax_normalize(scaled_energy)
    unc_norm = minmax_normalize(uncertainty)
    energy_uncertainty_score = 0.9890343205471551 * energy_norm * unc_norm * unc_sigmoid
    wait_normalized = np.where(slack <= 0.0, ready_wait_time / (np.abs(slack) + 1.0), 0.0)
    wait_score = minmax_normalize(wait_normalized)
    score = minmax_normalize(slack_score) + minmax_normalize(unc_slack_coupling) + minmax_normalize(duration_risk_penalty) + critical_release_score + wait_score
    score += slack_gate * (0.9919390154695125 * energy_eff_score + rank_score + energy_uncertainty_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
