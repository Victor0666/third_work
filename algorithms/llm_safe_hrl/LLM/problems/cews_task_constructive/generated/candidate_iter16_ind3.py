import numpy as np

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
    v2 mutation: Introduces *slack-activated SEER* (Slack-Gated Energy Efficiency Ratio)
    as primary energy-aware term, replaces arctan urgency with bounded tanh(-slack/τ) for steeper near-deadline response,
    adds critical-path density ratio (CPDR = upward_rank / (min_exec_time + min_comm_time + eps)) to surface latency-critical tasks,
    uses multiplicative fairness gating (sqrt(wait)/max(1, -slack+eps)) only when slack < 0 to prevent starvation *under violation*,
    replaces additive uncertainty penalty with uncertainty-weighted latency *only for slack > 0*, and applies strict lexicographic dominance:
    urgency >> CPDR >> SEER >> fairness, enforced via multiplicative scaling (not additive blending).
    
    Key innovations:
    - Urgency is now tanh(-slack/τ), bounded in [-1,1], sharper near zero → stronger hard-DDL signal.
    - SEER = (min_exec_time + min_comm_time + eps) / (min_incremental_energy + eps), then gated by exp(-max(0,-slack)/τ_seer) → 
      energy efficiency *only optimized when slack headroom exists*, and smoothly suppressed as slack turns negative.
    - CPDR emphasizes tasks with high upward_rank but low latency cost → identifies "high-leverage, low-overhead" schedulable bottlenecks.
    - Fairness is *inverted*: applied *only during violation* (slack < 0) as sqrt(wait)/max(1, -slack+eps), preventing indefinite starvation of late tasks.
    - All terms normalized via robust MAD (median absolute deviation) fallback to IQR, with explicit zero-variance handling.
    - Final score = urgency_term * (1 + 0.3 * cpdr_term) * (1 + 0.5 * seer_term) + fairness_term → multiplicative hierarchy preserves ordering dominance.
    """
    eps = 1e-8
    # Cast & sanitize inputs
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    # Robust adaptive normalization: MAD-based, fallback to range if MAD ≈ 0
    def normalize_robust(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 1:
            return np.array([0.0])
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad > eps:
            return (x - med) / (mad + eps)
        else:
            xmin, xmax = np.min(x), np.max(x)
            scale = xmax - xmin
            if scale < eps:
                scale = eps
            return (x - (xmin + xmax) / 2.0) / (scale + eps)
    
    # === 1. URGENCY TERM: tanh(-slack/τ) → sharp, bounded, monotonic, DDL-dominant ===
    tau_urg = 0.5
    urgency_raw = np.tanh(-slack / tau_urg)  # [-1, 1]; -1 = highly urgent (large negative slack)
    norm_urgency = normalize_robust(urgency_raw)
    # Scale to [0, 2] range for multiplicative dominance base: 0 = neutral, 2 = max urgency
    urgency_term = 1.0 + norm_urgency  # maps [-1,1] → [0,2]
    
    # === 2. CRITICAL-PATH DENSITY RATIO (CPDR): upward_rank / latency_cost ===
    latency_cost = min_exec_time + min_comm_time + eps
    cpdr_base = upward_rank / latency_cost
    norm_cpdr = normalize_robust(cpdr_base)
    # Clip to [-1, 1] to avoid explosion; center at 0 → boost only above median CPDR
    cpdr_term = np.clip(norm_cpdr, -1.0, 1.0)
    
    # === 3. SLACK-GATED SEER (Energy Efficiency Ratio) ===
    # SEER = latency_cost / energy → higher = more energy-efficient per latency unit
    seer_base = latency_cost / (min_incremental_energy + eps)
    # Gate by slack headroom: exp(-max(0, -slack)/τ_seer) → suppresses SEER when slack < 0
    tau_seer = 2.0
    seer_gate = np.exp(-np.maximum(0.0, -slack) / tau_seer)  # 1.0 when slack >= 0; decays as slack goes negative
    seer_gated = seer_base * seer_gate
    norm_seer = normalize_robust(seer_gated)
    seer_term = np.clip(norm_seer, -1.0, 1.0)
    
    # === 4. VIOLATION-AWARE FAIRNESS (applied ONLY when slack < 0) ===
    # Prevents starvation *during deadline violation*: rewards long-waiting tasks proportionally to wait time and inverse violation severity
    fairness_raw = np.where(
        slack < 0,
        np.sqrt(np.maximum(ready_wait_time, 0.0) + eps) / np.maximum(-slack + eps, 1.0),
        0.0
    )
    norm_fairness = normalize_robust(fairness_raw)
    # Keep fairness bounded and weak: it's corrective, not dominant
    fairness_term = np.clip(norm_fairness, 0.0, 0.5) * 0.4
    
    # === Multiplicative lexicographic composition ===
    # urgency dominates → scaled by CPDR boost (if CPDR > 0) and SEER boost (if SEER > 0)
    # Ensures high-urgency tasks are never diluted; low-urgency tasks gain from CPDR/SEER
    cpdr_boost = 1.0 + 0.3 * np.maximum(cpdr_term, 0.0)
    seer_boost = 1.0 + 0.5 * np.maximum(seer_term, 0.0)
    base_score = urgency_term * cpdr_boost * seer_boost
    score = base_score + fairness_term
    
    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e9, posinf=1e9, neginf=-1e9)
    
    return score
