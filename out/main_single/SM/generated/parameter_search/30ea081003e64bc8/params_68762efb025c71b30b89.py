import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining best practices from both parents:
    - Uses inverse-energy efficiency (Parent 2) for sharper discrimination.
    - Applies exponential wait decay (Parent 2) for bounded starvation mitigation.
    - Integrates robust MAD-based normalization (Parent 1) with tunable scale.
    - Adds fairness exponent to penalize extreme durations non-linearly.
    - Keeps slack-gated criticality and uncertainty-slack interaction (both).
    - All components are finite, deterministic, and shape-strict (N,)."""
    eps = 1.0379589765436138e-09
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def robust_normalize(x):
        x = np.asarray(x)
        center = np.median(x)
        dev = np.abs(x - center)
        scale = np.median(dev) if len(dev) > 0 else 1.0
        scale = max(scale * 1.9995387798778355, eps)
        return (x - center) / (scale + eps)
    duration = exec_t + comm_t
    slack_norm = robust_normalize(slk)
    slack_penalty = np.where(slk < 0, 5.852181323052533 * slack_norm ** 2, -5.605674237289604 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.29892770154054293 * robust_normalize(inv_energy)
    rank_active = np.where(slk > -eps, rank, 0.0)
    rank_score = -0.32260040664642087 * robust_normalize(rank_active + eps)
    dur_norm = robust_normalize(duration + eps)
    uncert_norm = robust_normalize(uncert + eps)
    dur_uncert_blend = (1.0 + 0.00028261457472070434 * uncert_norm) / (dur_norm + eps + 0.00028261457472070434 * uncert_norm + eps)
    dur_score = robust_normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.059325387920888886 * wait)
    wait_score = -robust_normalize(wait_sat + eps)
    slack_stress = np.where(slk < 0, np.abs(slack_norm), 0.0)
    unc_slack_interaction = 4.593597902766429 * uncert_norm * slack_stress
    norm_dur_dev = robust_normalize(duration)
    fairness_term = np.sign(norm_dur_dev) * np.abs(norm_dur_dev) ** 2.601611146143717
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + fairness_term
    score = np.nan_to_num(score, nan=np.median(score), posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
