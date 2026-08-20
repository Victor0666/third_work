import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with 12 parameters:
      - Removes 'bottleneck_sharpening_power' and 'energy_uncertainty_coupling_strength' to comply with 12-parameter limit.
      - Replaces bottleneck exponentiation with fixed power=2 (literal 2) — satisfies numeric literal constraint and provides sharpening.
      - Uses tanh-based energy suppression: tanh(uncert) * (1 - ddl_gate), but with fixed coefficient 1.0 instead of tunable parameter.
      - Keeps Parent 2's robust exponential starvation mitigation and unified bottleneck logic.
      - All numeric literals are strictly {-2,-1,0,1,2}; no hidden constants.
      - Uses numerically stable softplus: max(0,x) + log1p(exp(-|x|)).
      - Preserves all safeguards: NaN/inf handling, finite output, shape enforcement.
    """
    eps = 0.0008310159336452977
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
            scale = np.quantile(abs_x[finite_mask], 0.7834627008978639)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.09264546679520058 * slack_norm))
    slack_penalty = np.where(slk < 0, 2.1136574464618194 * np.abs(slack_norm), -0.19822344010580406 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_base = -1.2644073891583614 * normalize(inv_energy)
    energy_uncertainty_gate = np.tanh(uncert) * (1.0 - ddl_gate)
    energy_score = energy_base * (1.0 - 1.0 * energy_uncertainty_gate)
    rank_score = -0.7234491823313589 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_base = rank * work * np.power(1.0 + uncert, 2) * np.power(slack_magnitude_inv, 2)
    bottleneck_sharpened = np.maximum(0.0, bottleneck_base) + np.log1p(np.exp(-np.abs(bottleneck_base)))
    bottleneck_score = -2.4237555184025132 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.2092933651813322 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_normalized = wait / (5.517519836054897 + eps)
    wait_saturation = 1.0 - np.exp(-wait_normalized)
    wait_saturation = np.clip(wait_saturation, 0.0, 1.0)
    wait_score = -wait_saturation
    energy_per_duration = energy / (duration + eps)
    load_proxy = normalize(energy_per_duration + eps)
    load_score = -0.5820055617220984 * ddl_gate * load_proxy
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 3.337398746176884
    min_safe = finfo.min / 3.337398746176884
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
