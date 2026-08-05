import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Novel priority rule emphasizing deadline risk nonlinearity, critical-path balance,
    energy efficiency per work, and saturation-based anti-starvation.
    
    Key innovations:
      - Slack penalty uses clipped power law: (max(0, -slack))^exponent → strong but bounded urgency
      - Critical path boost applies only above median rank → avoids over-prioritizing all high-rank tasks
      - Energy efficiency ratio = energy / (work + eps) → favors low-energy-per-MI tasks
      - Duration uncertainty penalty = std(duration)/mean(duration+eps) → penalizes volatile execution/comm
      - Ready wait uses smooth saturation: 1 - exp(-wait/threshold) → anti-starvation with diminishing returns
      - Rank gap term promotes diversity: prioritizes tasks *below* max rank to avoid congestion on single path
    """
    eps = 0.0003295324196489967
    N = len(min_exec_time)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_normalize(x):
        x_abs = np.abs(x)
        x_mean = np.mean(x_abs)
        scale = x_mean + eps
        return x / scale
    neg_slack = np.maximum(-slack, 0.0)
    slack_risk = np.power(neg_slack + eps, 2.358131613529369)
    slack_risk_norm = robust_normalize(slack_risk)
    rank_median = np.median(upward_rank)
    rank_boost_mask = (upward_rank >= rank_median).astype(float)
    rank_boosted = upward_rank * (1.0 + 1.9496489111718134 * rank_boost_mask)
    rank_score = -robust_normalize(rank_boosted)
    energy_per_work = min_incremental_energy / (remaining_work + eps)
    energy_efficiency_score = robust_normalize(energy_per_work)
    duration = min_exec_time + min_comm_time
    duration_mean = np.mean(duration) + eps
    duration_std = np.std(duration)
    duration_uncert_ratio = duration_std / duration_mean
    duration_uncert_penalty = 0.8205348845176695 * np.full(N, duration_uncert_ratio)
    wait_saturation = 1.0 - np.exp(-ready_wait_time / (1.1012076253905576 + eps))
    wait_score = -robust_normalize(wait_saturation)
    slack_pressure_indicator = (neg_slack > eps).astype(float)
    coupled_risk = uncertainty * slack_pressure_indicator
    coupled_risk_norm = robust_normalize(coupled_risk)
    rank_max = np.max(upward_rank) + eps
    rank_gap = (rank_max - upward_rank) / rank_max
    rank_gap_score = -robust_normalize(rank_gap) * 1.844869761325997
    score = slack_risk_norm * 1.0 + energy_efficiency_score * 1.2039795250061809 + rank_score * 1.0 + duration_uncert_penalty * 1.0 + wait_score * 1.0 + coupled_risk_norm * 1.3641692778353078 + rank_gap_score * 1.0
    score = np.nan_to_num(score, nan=877799.9209503441, posinf=16379.879403958394, neginf=-2267329.8681353573)
    return score
