import numpy as np

# 函数名中的 2 由 SeEvo 在生成 Prompt 时替换为目标版本号。
# 八个输入均为长度 N 的一维数组；相同下标始终指向同一个 ready task。
# 返回值也必须是长度 N 的一维数组，并遵守“分数越小，优先级越高”。
def get_task_priority_v2(
    min_exec_time,
    min_comm_time,
    min_incremental_energy,
    slack,
    upward_rank,
    remaining_work,
    ready_wait_time,
    uncertainty
):

    """
    Self-evolved priority rule: deadline-criticality dominance + uncertainty-aware harmonic efficiency +
    starvation-safe wait boosting + slack-relative work penalty + latency-energy decoupling.

    Key evolutions from v1:
      - Replaces tanh urgency with *hard-gated linear ramp* (0→1) over [slack_30, slack_30 - duration], 
        ensuring monotonic, interpretable, and numerically stable urgency signal — eliminates tanh saturation artifacts.
      - Introduces *uncertainty-weighted harmonic efficiency*: harmonic_ce * (1 - clip(uncertainty, 0, 1)), 
        downweighting efficiency scores for high-uncertainty tasks — aligns with risk-first objective.
      - Starvation guard now *duration-normalized*: boost = (ready_wait_time - p75) / (duration + eps), 
        preventing artificial dominance by long-duration tasks in low-load scenarios.
      - Work penalty uses *relative slack depletion*: slack_gap / (|median_slack| + eps), not |slack_30|, 
        improving robustness when slack distribution is skewed or sparse.
      - Decouples latency and energy scoring: latency_score now *inverted and scaled by urgency*, 
        so only urgent tasks pay latency cost — avoids penalizing low-latency non-urgent tasks.
      - All scaling uses 'higher-moment fallback': if IQR ≈ 0, use std instead of MAD to preserve dispersion signal 
        under tight distributions (e.g., identical slack values).
      - Final score enforces strict priority ordering: urgency term dominates (weight=3.0), others strictly subordinated.
      - Explicit finite-range enforcement: all intermediate terms clipped before combination.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_scale(x):
        if len(x) == 0:
            return x
        med = np.median(x)
        x_centered = x - med
        q1, q3 = np.quantile(x_centered, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        if iqr < eps:
            # Prefer std over MAD for tighter distributions to retain relative spread
            std_val = np.std(x_centered, ddof=0)
            scale = std_val if std_val > eps else np.mean(np.abs(x_centered)) + eps
        else:
            scale = iqr + eps
        scaled = x_centered / scale
        return np.clip(scaled, -1e6, 1e6)

    duration = np.maximum(min_exec_time + min_comm_time, eps)
    ce_ratio = upward_rank / (min_incremental_energy + eps)
    synergy_ratio = upward_rank * duration / (min_incremental_energy + eps)
    harmonic_ce = 2.0 / (1.0 / (ce_ratio + eps) + 1.0 / (synergy_ratio + eps))
    harmonic_ce = np.clip(harmonic_ce, eps, 1e6)
    
    # Uncertainty-aware efficiency: suppress score for high-uncertainty tasks
    uncertainty_weight = 1.0 - np.clip(uncertainty, 0.0, 1.0)
    weighted_harmonic_ce = harmonic_ce * uncertainty_weight

    # Hard-gated linear urgency: ramps from 0 (at slack_30) to 1 (at slack_30 - duration), clamped
    slack_30 = np.quantile(slack, 0.3, method='midpoint') if N > 1 else np.min(slack)
    urgency_base = np.clip((slack_30 - slack) / (duration + eps), 0.0, 1.0)
    urgency_score = np.where(slack < slack_30, urgency_base, 0.0)
    urgency_score = np.clip(urgency_score, 0.0, 1.0)

    # Duration-normalized starvation boost: activates only for long-waiting *and* low-uncertainty tasks
    wait_thresh = np.quantile(ready_wait_time, 0.75) if N > 1 else np.max(ready_wait_time)
    starvation_cond = (ready_wait_time > wait_thresh + eps) & (uncertainty < 0.75)
    wait_boost = np.where(
        starvation_cond,
        (ready_wait_time - wait_thresh) / (duration + eps),
        0.0
    )
    wait_score = -np.clip(wait_boost, 0.0, 1e3)

    # Relative slack depletion: uses median slack for better skew robustness
    median_slack = np.median(slack)
    slack_gap = np.clip(slack_30 - slack, 0.0, None)
    work_norm = robust_scale(remaining_work)
    work_penalty = work_norm * (slack_gap / (np.abs(median_slack) + eps))

    # Energy score: risk-adjusted but capped and scaled
    energy_base = min_incremental_energy * (1.0 + 0.8 * np.clip(uncertainty, 0.0, 1.0))
    energy_score = robust_scale(energy_base)

    # Latency score: only penalizes latency *when urgent*, inverted for priority
    latency_score = robust_scale(duration)
    latency_score = np.where(urgency_score > 0.1, -latency_score, 0.0)

    # Normalize key components
    urgency_norm = robust_scale(urgency_score)
    ce_norm = robust_scale(weighted_harmonic_ce)
    ce_score = -ce_norm
    synergy_norm = robust_scale(synergy_ratio)
    synergy_score = -synergy_norm

    # Dominant urgency term ensures hard DDL compliance; others are secondary tradeoffs
    score = (
        3.0 * -urgency_norm +           # Highest weight: strict deadline enforcement
        0.35 * ce_score +              # Criticality-energy efficiency (uncertainty-dampened)
        0.25 * synergy_score +         # Latency-synergy balance
        0.2 * energy_score +           # Risk-adjusted marginal energy
        0.15 * latency_score +         # Urgency-conditioned latency preference
        0.12 * wait_score +            # Starvation guard (duration-normalized)
        0.18 * work_penalty            # Heavy-subtree penalty only under slack pressure
    )

    # Final safeguard: ensure finite, deterministic, shape-correct output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
