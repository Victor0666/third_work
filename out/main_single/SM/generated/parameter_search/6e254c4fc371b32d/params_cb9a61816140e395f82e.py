import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: DDL-hard constraint first, then energy minimization.
    
    Key structural improvements:
      - Replaces double normalization with single, stable `safe_normalize` using mean-abs + epsilon.
      - Uses bounded slack-criticality gate: criticality suppressed only when slack is severely negative.
      - Energy term uses raw `1/(energy + eps)` — preserves relative discrimination without destructive rescaling.
      - Introduces fairness power-law on duration: `sign(duration_dev) * |duration_dev|^p` for asymmetric bias control.
      - Uncertainty-slack interaction is strictly multiplicative *only* under stress (`slack < 0`).
      - Wait-time uses saturating exponential: smooth, bounded, and avoids runaway priority.
    """
    eps = 1.3010596499744243e-05
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def safe_normalize(x):
        x = np.asarray(x)
        scale = np.mean(np.abs(x)) + eps
        return x / scale
    duration = exec_t + comm_t
    dur_center = np.mean(duration)
    dur_dev = duration - dur_center
    fairness_term = np.sign(dur_dev) * np.abs(dur_dev) ** 1.047992182598783
    norm_fairness = safe_normalize(fairness_term + eps)
    pos_slack = slk[slk > eps]
    slack_margin = np.median(pos_slack) if len(pos_slack) > 0 else 12.077702139411315
    slack_margin = max(slack_margin, eps)
    slack_penalty = np.where(slk < 0, 4.356463762857974 * (np.abs(slk) / (slack_margin + eps)) ** 2, -1.2278318804239134 * (np.abs(slk) / (slack_margin + eps)))
    gate = np.clip(1.0 - np.maximum(-slk - slack_margin, 0.0) / (slack_margin + eps), 0.0, 1.0)
    gated_rank = rank * gate
    rank_score = -0.022015966987335614 * safe_normalize(gated_rank + eps)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.1144584518663574 * inv_energy
    unc_slack_interaction = np.where(slk < 0, 1.451473720830343 * uncert * (np.abs(slk) / (slack_margin + eps)), 0.0)
    wait_sat = 1.0 - np.exp(-2.8266302833906152e-05 * wait)
    wait_score = -safe_normalize(wait_sat + eps)
    score = slack_penalty + energy_score + rank_score + norm_fairness + wait_score + unc_slack_interaction
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=finfo.tiny, posinf=finfo.max / 305.7578257085864, neginf=-finfo.max / 305.7578257085864)
    assert score.shape == (N,), f'Expected (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values remain after sanitization'
    return score
