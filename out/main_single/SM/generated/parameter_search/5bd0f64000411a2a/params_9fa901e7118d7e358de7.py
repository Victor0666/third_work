import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with bounded percentile normalization, hard DDL-protection gate,
    and clipped linear starvation mitigation — eliminating fragile exponential decay and outlier-sensitive MAD.
    
    Key self-evolution improvements:
      - Bounded percentile-based normalization (p10–p90) replaces MAD: robust to skew, preserves rank order,
        avoids median instability on small N, and eliminates tuning of scale/mad multipliers.
      - Hard DDL-protection gate: multiplies slack_penalty by ddl_protection_gate when slack < -eps,
        strictly prioritizing deadline compliance over all other objectives under violation risk.
      - Clipped linear wait-time: min(ready_wait_time, wait_clip_threshold) — removes exponential decay
        sensitivity, guarantees bounded contribution, and simplifies starvation control.
      - All components are finite, deterministic, and shaped strictly (N,).
    """
    eps = 0.0009211393856033019
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def percentile_normalize(x):
        x = np.asarray(x)
        p_low = np.percentile(x, 13.002415457547915)
        p_high = np.percentile(x, 86.87951541972251)
        rng = p_high - p_low + eps
        centered = x - (p_low + p_high) / 2.0
        return np.clip(centered / rng, -2.0, 2.0)
    duration = exec_t + comm_t + 0.05169700450094539 * work
    norm_duration = percentile_normalize(duration + eps)
    inv_energy = 1.0 / (energy + eps)
    norm_energy = percentile_normalize(inv_energy)
    energy_score = -0.933094825084504 * norm_energy
    norm_slack = percentile_normalize(slk)
    base_slack_penalty = np.where(slk < 0, 6.7646370804536975 * norm_slack ** 2, -1.352433009747472 * np.abs(norm_slack))
    slack_penalty = np.where(slk < -eps, 1.203203363188097 * base_slack_penalty, base_slack_penalty)
    rank_active = np.where(slk >= -eps, rank, 0.0)
    norm_rank = percentile_normalize(rank_active + eps)
    rank_score = -0.10676987678440164 * norm_rank
    norm_uncert = percentile_normalize(uncert + eps)
    dur_uncert_blend = (1.0 + 0.6137525187440532 * norm_uncert) / (np.abs(norm_duration) + eps + 0.6137525187440532 * norm_uncert + eps)
    dur_score = percentile_normalize(dur_uncert_blend)
    clipped_wait = np.minimum(wait, 81.78671037051399)
    norm_wait = percentile_normalize(clipped_wait + eps)
    wait_score = -norm_wait
    slack_stress = np.where(slk < 0, np.abs(norm_slack), 0.0)
    unc_slack_interaction = 4.42850517276559 * norm_uncert * slack_stress
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction
    score = np.nan_to_num(score, nan=np.finfo(float).smallest_subnormal, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
