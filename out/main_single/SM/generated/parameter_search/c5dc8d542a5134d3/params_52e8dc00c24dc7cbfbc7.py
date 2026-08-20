import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Novel priority rule emphasizing robust DDL compliance first, then energy efficiency.
    
    Key innovations:
      - Piecewise slack handling: strong quadratic penalty for negative slack, soft gain for positive slack.
      - Inverse energy efficiency: 1/(energy + epsilon) — favors low marginal energy more sharply.
      - Criticality gated by slack: upward_rank only contributes meaningfully when slack > 0.
      - Uncertainty-slack interaction: multiplicative boost to uncertainty penalty under deadline stress.
      - Wait-time modeled with exponential decay: prioritizes long-waiting tasks but saturates smoothly.
      - Duration-uncertainty ratio: combines exec+comm with uncertainty via weighted harmonic mean for stability.
    """
    eps = 0.0018970922801918358
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def safe_mean_abs(x):
        m = np.mean(np.abs(x))
        return m if np.isfinite(m) and m > 0 else eps

    def normalize(x):
        x = np.asarray(x)
        return x / (safe_mean_abs(x) + eps)
    slack_norm = normalize(slk)
    slack_penalty = np.where(slk < 0, 1.9984255309430827 * slack_norm ** 2, -5.380513111034151 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.6616809410952393 * normalize(inv_energy)
    rank_active = np.where(slk > -eps, rank, 0.0)
    rank_score = -0.05029031855651997 * normalize(rank_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = (1.0 + 0.40772825653857403 * uncert_norm) / (dur_norm + eps + 0.40772825653857403 * uncert_norm + eps)
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.0005389952551190192 * wait)
    wait_score = -normalize(wait_sat + eps)
    slack_stress = np.where(slk < 0, np.abs(slack_norm), 0.0)
    unc_slack_interaction = 2.646577597948439 * uncert_norm * slack_stress
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
