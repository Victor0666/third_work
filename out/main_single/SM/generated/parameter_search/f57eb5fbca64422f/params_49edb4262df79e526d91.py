import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Keeps Parent 2's robust sigmoid ddl_protection_gate and slack_urgency_gain.
      - Adds Parent 1's *additive* slack-coupled energy modulation (not multiplicative) to avoid blowup near zero slack.
      - Retains Parent 2's simplified linear duration-uncertainty blend but augments with Parent 1's starvation-aware wait_ramp.
      - Introduces new 'energy_slack_coupling_strength' via reuse of existing 'ddl_protection_gate' — avoids adding parameter.
        Specifically: use PARAMS["ddl_protection_gate"] as coupling strength (same semantic role: controls sensitivity near slack=0).
      - All normalizations use same robust quantile-based scheme; no hard clipping or median bias.
      - Strict DDL-first enforced via dominant slack_penalty term.
    """
    eps = 0.07887781685028741
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], 0.7360724488067147)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.3787840094061239 * slack_norm))
    slack_penalty = np.where(slk < 0, 4.473446233478846 * np.abs(slack_norm), -0.7684380123677172 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    base_energy_score = -0.12110930256524349 * normalize(inv_energy)
    slack_proximity = np.clip(1.0 - np.abs(slack_norm), 0.0, 1.0)
    energy_score = base_energy_score * ddl_gate + 0.3787840094061239 * base_energy_score * slack_proximity
    rank_score = -0.17601773277782015 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 3.150969668857762)
    bottleneck_score = -1.7149474145933474 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.843421660302451 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 6.025082963405983)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 7438453.1844911855
    min_safe = -finfo.max / 7438453.1844911855
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
