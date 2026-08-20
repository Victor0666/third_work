import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with:
      - Replaced exponential wait decay with bounded linear clipping (more stable & interpretable).
      - Added conditional DDL protection gate: activates criticality only when slack is *very tight*.
      - Introduced load-successor-release interaction: upward_rank * remaining_work, gated by negative slack.
      - Removed inactive uncertainty-slack interaction per evidence; replaced with robust bottleneck-aware term.
      - All normalizations use median-based scaling (more outlier-resilient than mean-abs).
      - Strict slack-first feasibility enforcement via sign-aware gating.
    """
    eps = 0.0008768245426891596
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
        med = np.median(abs_x[np.isfinite(abs_x)])
        scale = med if np.isfinite(med) and med > eps else eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    slack_penalty = np.where(slk < 0, 5.8841979776738516 * np.abs(slack_norm), np.where(slk <= 0.0, 0.0, -3.6331438454848777 * np.abs(slack_norm)))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.7518911211077142 * normalize(inv_energy)
    rank_active = np.where(slack_norm <= 0.450549040122867, rank, 0.0)
    rank_score = -2.6880377511472853 * normalize(rank_active + eps)
    bottleneck_boost = rank * work
    bottleneck_active = np.where(slk < 0, bottleneck_boost, 0.0)
    bottleneck_score = -1.591626031474396 * normalize(bottleneck_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.5377067628860343 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 1.3026446770369302)
    wait_score = -normalize(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 1318996.3147277453
    min_safe = -finfo.max / 1318996.3147277453
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
