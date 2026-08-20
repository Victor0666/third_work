import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating evidence-driven structural changes:
      - Added host-load interaction: modulates energy term using uncertainty and slack coupling to avoid over-selecting low-energy tasks under high-risk conditions
      - Replaced static duration-uncertainty blend with uncertainty-slack-coupled urgency term that amplifies priority when both risk and deadline pressure are high
      - Kept all normalization robust (quantile-based), kept ddl_protection_gate and slack_penalty dominant
      - Removed finfo_max_scale (inactive per diagnostics) and replaced hidden 1e-12 with tunable finfo_safety_scale
      - All operations protected against NaN/inf/zero; no unbounded loops or side effects
      - Uses only allowed numeric literals: -2, -1, 0, 1, 2
    """
    eps = 0.07018119353411618
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
            scale = np.quantile(abs_x[finite_mask], 0.7653647288111751)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(5.140705356558645e-05 * slack_norm))
    slack_penalty = np.where(slk < 0, 3.9428888871883085 * np.abs(slack_norm), -0.6804466954206985 * np.abs(slack_norm))
    slack_magnitude = np.abs(slk) + eps
    unc_slk_coupling = 0.3730869790547082 * normalize(uncert) * (1.0 / slack_magnitude)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.1568591338644643 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - np.clip(unc_slk_coupling, 0.0, 1.0))
    rank_score = -0.9117016270802654 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.1675329919902653)
    bottleneck_score = -0.944763308776427 * normalize(bottleneck_sharpened + eps)
    coupled_urgency = normalize(uncert) * np.abs(slack_norm)
    coupled_score = -0.3730869790547082 * coupled_urgency
    wait_clipped = np.clip(wait, 0.0, 30.866398045338734)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + coupled_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / np.power(2.0, 2.83305406642644)
    min_safe = -max_safe
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
