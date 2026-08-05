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
    Self-evolved priority rule: strict deadline enforcement first, then crisp criticality-energy tradeoff,
    starvation-aware wait boosting with adaptive thresholding, and slack-proportional work penalty with hard gating.
    
    Key evolutions from v1:
      - Drops joint feature scaling → restores per-feature robust scaling to preserve urgency signal integrity;
        uses independent median/IQR per term for fidelity to physical semantics (e.g., slack sign matters).
      - Replaces sigmoid slack gating with *pure percentile-30 urgency threshold* + *hard penalty amplification*
        only for tasks strictly below that threshold — eliminates soft decay artifacts harming DDL compliance.
      - Removes uncertainty damping on ce_ratio → lets high-uncertainty critical tasks dominate scheduling
        when deadlines are tight (aligns with risk-adjusted objective: penalize lateness *first*, then optimize energy).
      - Introduces *adaptive wait-threshold*: uses 75th percentile of ready_wait_time to trigger starvation boost,
        more discriminative than median and avoids premature activation in sparse/low-load scenarios.
      - Replaces sigmoid work penalty with *hard-gated linear slack depletion*: work_penalty = work_norm * max(0, 1 - slack / (|slack_30| + eps)),
        ensuring heavy subtrees are penalized *only when urgent*, not merely "less slack".
      - Adds *latency-criticality synergy*: (upward_rank * duration) / (min_incremental_energy + eps), normalized separately,
        to prioritize high-importance, low-energy-per-latency tasks — balances speed and efficiency without noise coupling.
      - All terms bounded, epsilon-safe, finite-preserving, deterministic, and explicitly shape-checked.
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

    # Robust scaling per feature (independent, sign-preserving, median-centered)
    def robust_scale(x):
        if len(x) == 0:
            return x
        med = np.median(x)
        x_centered = x - med
        q1, q3 = np.quantile(x_centered, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1
        if iqr < eps:
            mad = np.median(np.abs(x_centered))
            scale = mad if mad > eps else np.mean(np.abs(x_centered)) + eps
        else:
            scale = iqr + eps
        return np.clip(x_centered / scale + med, -1e6, 1e6)

    # Critical base terms
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    neg_slack = np.maximum(-slack, 0.0)

    # --- Deadline Enforcement Block (highest weight, no softening) ---
    slack_sorted = np.sort(slack)
    slack_30 = slack_sorted[max(0, int(0.3 * len(slack_sorted)))] if N > 0 else 0.0
    is_deeply_urgent = slack < slack_30  # Strict below 30th percentile → highest priority
    # Amplified penalty: base penalty + fixed bonus for deepest urgency
    deadline_penalty = neg_slack * (6.0 + 4.0 * np.clip(uncertainty, 0.0, 2.0))
    deadline_penalty = np.where(is_deeply_urgent, deadline_penalty + 12.0, deadline_penalty)

    # --- Criticality-Energy Tradeoff (no uncertainty damping) ---
    ce_ratio = upward_rank / (min_incremental_energy + eps)
    ce_score = -robust_scale(ce_ratio)  # Higher upward_rank / lower energy → lower score → higher priority

    # --- Latency-Criticality Synergy: favors high-rank, low-duration, low-energy tasks ---
    synergy = (upward_rank * duration) / (min_incremental_energy + eps)
    synergy_score = -robust_scale(synergy)

    # --- Energy Base (risk-adjusted) ---
    energy_base = min_incremental_energy * (1.0 + 0.7 * np.clip(uncertainty, 0.0, 1.0))
    energy_score = robust_scale(energy_base)

    # --- Latency Score ---
    latency_score = robust_scale(duration)

    # --- Starvation Mitigation: adaptive wait-threshold at 75th percentile ---
    wait_thresh = np.quantile(ready_wait_time, 0.75) if N > 0 else eps
    wait_boost = np.where(ready_wait_time > wait_thresh, 
                          1.0 * (ready_wait_time - wait_thresh) / (np.maximum(np.std(ready_wait_time), eps) + eps),
                          0.0)
    wait_score = -wait_boost  # Negative boost → lowers score for starved tasks

    # --- Slack-Proportional Work Penalty: hard-gated linear depletion ---
    slack_gap = np.clip(slack_30 - slack, 0.0, None)  # Only active when slack < slack_30
    work_norm = robust_scale(remaining_work)
    work_penalty = work_norm * (slack_gap / (np.abs(slack_30) + eps))

    # --- Final weighted aggregation ---
    score = (
        1.8 * deadline_penalty +     # Strongest weight: enforce hard DDL
        0.2 * latency_score +      # Prefer fast-executing tasks when safe
        0.3 * energy_score +       # Lower risk-adjusted energy when feasible
        0.45 * ce_score +          # Criticality-energy ratio dominates non-urgent tier
        0.35 * synergy_score +     # Prioritizes latency-efficient critical work
        0.12 * wait_score +        # Modest starvation relief, gated and scaled
        0.18 * work_penalty        # Penalizes heavy subtrees *only* under urgency
    )

    # Final sanitization: ensure finite, shape-(N,), deterministic
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    assert score.shape == (N,), f"Expected shape (N,)={N}, got {score.shape}"
    return score
