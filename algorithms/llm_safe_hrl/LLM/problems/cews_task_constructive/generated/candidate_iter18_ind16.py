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

    '''
    Priority rule v2 (self-evolved): Restores urgency dominance & energy sensitivity while fixing signal distortion.
    Key improvements:
      - Urgency term uses wider dynamic range (0.05–10.0) and preserves monotonicity near deadline (arctan → softplus for better gradient)
      - SEER term: removed aggressive clipping; uses robust z-score fallback when MAD degenerates, preserving relative energy/latency ranking
      - Risk-criticality: keeps tanh(uncertainty) boost but normalizes *before* multiplication to avoid scale explosion
      - Aging: replaces tanh with scaled sigmoid + linear ramp to ensure monotonic fairness under safety margin
      - All pre-normalization now preserves original signal semantics: no clamping of uncertainty or energy; uses eps-avoidance only
      - Final score combines urgency-gated SEER multiplicatively (not additively) to enforce lexicographic ordering
      - Strict finite output via np.clip + nan_to_num with conservative bounds
    '''
    eps = 1e-08
    # Safe casting without semantic distortion: preserve original magnitude & sign
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    # Robust NaN/inf handling *after* casting — no premature clamping
    min_exec_time = np.nan_to_num(min_exec_time, nan=eps, posinf=1e6, neginf=eps)
    min_comm_time = np.nan_to_num(min_comm_time, nan=eps, posinf=1e6, neginf=eps)
    min_incremental_energy = np.nan_to_num(min_incremental_energy, nan=eps, posinf=1e6, neginf=eps)
    slack = np.nan_to_num(slack, nan=0.0, posinf=1e6, neginf=-1e6)
    upward_rank = np.nan_to_num(upward_rank, nan=eps, posinf=1e6, neginf=eps)
    remaining_work = np.nan_to_num(remaining_work, nan=eps, posinf=1e6, neginf=eps)
    ready_wait_time = np.nan_to_num(ready_wait_time, nan=0.0, posinf=1e6, neginf=eps)
    uncertainty = np.nan_to_num(uncertainty, nan=0.0, posinf=1e2, neginf=0.0)
    
    def normalize_mad(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad < eps:
            # Fallback to z-score on variance (not MAD) if MAD vanishes — avoids zero-division & preserves ordering
            var = np.var(x)
            if var < eps:
                return np.zeros_like(x, dtype=float)
            return (x - med) / (np.sqrt(var) + eps)
        return (x - med) / (mad + eps)
    
    # Urgency: softplus(-slack) — smooth, monotonic, unbounded below → stronger discrimination near deadline
    # Bounded only at output to prevent overflow; retains full gradient where it matters most
    urgency_raw = np.log1p(np.exp(-slack))  # softplus(-slack) ≈ max(0,-slack) + log(1+exp(-|slack|))
    norm_urgency = normalize_mad(urgency_raw)
    urgency_term = np.clip(1.0 + 6.0 * norm_urgency, 0.05, 10.0)  # Wider range restores deadline pressure sensitivity
    
    # SEER: energy-per-unit-latency, gated only when slack < 0 → energy optimization suppressed *only* under violation
    seer_base = (min_incremental_energy + eps) / (min_exec_time + min_comm_time + eps)
    slack_gate = np.where(slack < 0, 0.0, 1.0)  # Hard gate: zero SEER influence when violating
    seer_gated = seer_base * slack_gate
    norm_seer = normalize_mad(seer_gated)
    seer_term = 1.0 - 0.6 * norm_seer  # No hard clip → preserves relative ranking among safe tasks
    
    # Risk-augmented criticality: normalize components *before* multiplication to avoid scale inflation
    norm_upward = normalize_mad(upward_rank)
    norm_work = normalize_mad(remaining_work)
    risk_boost = 1.0 + np.tanh(uncertainty)  # Smooth [1,2] boost, preserves uncertainty semantics
    risk_criticality_base = (norm_upward + norm_work) * risk_boost  # Sum instead of product → stable dynamic range
    norm_criticality = normalize_mad(risk_criticality_base)
    risk_criticality_term = -0.9 * norm_criticality  # Slightly stronger weight for critical-path integrity
    
    # Aging: sigmoid ramp ensures monotonic fairness, saturates smoothly at 1.0 only when slack > 0 and wait is long
    # Linear ramp component ensures minimal aging pressure when slack < 0 (deadline crisis mode)
    wait_ratio = (ready_wait_time + eps) / (np.abs(slack) + 1.0 + eps)
    sigmoid_aging = 1.0 / (1.0 + np.exp(-wait_ratio + 2.0))  # Centered at wait_ratio=2 → activates after safety margin
    linear_aging = np.clip(ready_wait_time * 0.05, 0.0, 0.8)  # Gentle linear baseline for all tasks
    combined_aging = 0.7 * sigmoid_aging + 0.3 * linear_aging
    norm_aging = normalize_mad(combined_aging)
    aging_term = 0.18 * norm_aging  # Slightly increased weight to prevent starvation without compromising urgency
    
    # Lexicographic composition: urgency dominates multiplicatively; SEER only modulates *within* urgency tier
    # Criticality and aging added linearly for stability and interpretability
    score = urgency_term * (1.0 + 0.6 * seer_term) + risk_criticality_term + aging_term
    
    # Final safeguard: ensure finite, deterministic, bounded output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    score = np.clip(score, -1e9, 1e9)
    return score.astype(float)
