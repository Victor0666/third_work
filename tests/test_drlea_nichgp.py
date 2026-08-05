from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.comparisons.drlea_nichgp.action_mask import (
    masked_argmax_torch,
)
from algorithms.comparisons.drlea_nichgp.checkpointing import write_json
from algorithms.comparisons.drlea_nichgp.config import (
    AgentConfig,
    build_config,
    ensure_disjoint_seeds,
)
from algorithms.comparisons.drlea_nichgp.d3qn import MaskedD3QN
from algorithms.comparisons.drlea_nichgp.env_adapter import CEWSEnvAdapter
from algorithms.comparisons.drlea_nichgp.features import (
    routing_state,
    sequencing_state,
)
from algorithms.comparisons.drlea_nichgp.gp_features import (
    GP_TERMINALS,
    task_gp_terminals,
)
from algorithms.comparisons.drlea_nichgp.gp_primitives import (
    GPProgram,
    protected_division,
)
from algorithms.comparisons.drlea_nichgp.metrics import comparison_key
from algorithms.comparisons.drlea_nichgp.niching_gp import (
    behavior_characterization,
    load_rules,
)
from algorithms.comparisons.drlea_nichgp.replay_buffer import ReplayBatch
from algorithms.comparisons.drlea_nichgp.run_pipeline import run_pipeline


@pytest.fixture(scope="module")
def smoke_config():
    return build_config("SS", "T", 31, smoke=True)


@pytest.fixture(scope="module")
def adapter(smoke_config):
    value = CEWSEnvAdapter(smoke_config, 31)
    value.reset()
    return value


def test_instance_seed_and_fingerprint_reproducible(smoke_config):
    first = CEWSEnvAdapter(smoke_config, 31)
    second = CEWSEnvAdapter(smoke_config, 31)
    for adapter in (first, second):
        adapter.reset()
        while len(adapter.env.workflows) < 3:
            adapter.env.advance_to_next_resource_event()
    assert first.instance_descriptor() == second.instance_descriptor()
    assert first.instance_fingerprint() == second.instance_fingerprint()
    assert len(first.instance_descriptor()["workflows"]) == 3


def test_ra_sa_gp_dimensions_and_finiteness(adapter, smoke_config):
    task_id = adapter.advance_until_actionable()
    state, mask = routing_state(
        adapter, task_id, smoke_config.normalization
    )
    assert state.shape == (8 + 10 * adapter.num_vms,)
    assert mask.shape == (adapter.num_vms,)
    assert np.all(np.isfinite(state))
    assert sequencing_state(
        adapter, smoke_config.normalization
    ).shape == (16,)
    assert sequencing_state(
        adapter, smoke_config.normalization, []
    ).shape == (16,)
    assert sequencing_state(
        adapter, smoke_config.normalization, [task_id]
    ).shape == (16,)
    terminals = task_gp_terminals(
        adapter, task_id, smoke_config.normalization
    )
    assert terminals.shape == (14,)
    assert np.all(np.isfinite(terminals))


def test_mask_covers_random_greedy_and_double_target():
    config = AgentConfig(
        hidden_dims=(8, 8),
        replay_capacity=16,
        batch_size=2,
        epsilon_start=1.0,
        gamma=0.9,
    )
    agent = MaskedD3QN(3, 4, config, seed=7, device="cpu")
    mask = np.asarray([0, 1, 0, 1], dtype=np.float32)
    random_actions = {
        agent.choose_action(np.zeros(3), mask, training=True)[0]
        for _ in range(50)
    }
    assert random_actions <= {1, 3}
    agent.epsilon = 0.0
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        agent.online.advantage.bias.copy_(
            torch.tensor([100.0, 1.0, 90.0, 3.0])
        )
    action, kind = agent.choose_action(
        np.zeros(3), mask, training=False
    )
    assert (action, kind) == (3, "greedy_legal")

    with torch.no_grad():
        for parameter in agent.target.parameters():
            parameter.zero_()
        agent.target.advantage.bias.copy_(
            torch.tensor([10.0, 20.0, 30.0, 40.0])
        )
    batch = ReplayBatch(
        states=np.zeros((1, 3), dtype=np.float32),
        actions=np.asarray([1]),
        rewards=np.asarray([0.0], dtype=np.float32),
        next_states=np.zeros((1, 3), dtype=np.float32),
        dones=np.asarray([False]),
        masks=mask.reshape(1, -1),
        next_masks=mask.reshape(1, -1),
    )
    target = agent.compute_targets(batch)
    # Online picks legal action 3; target evaluates action 3.
    assert target.item() == pytest.approx(0.9 * 15.0)
    assert masked_argmax_torch(
        torch.tensor([[100.0, 1.0, 90.0, 3.0]]),
        torch.tensor(mask).reshape(1, -1),
    ).item() == 3


def test_nonterminal_zero_next_mask_rejected():
    agent = MaskedD3QN(
        2,
        2,
        AgentConfig(
            hidden_dims=(4, 4),
            replay_capacity=4,
            batch_size=1,
        ),
        seed=0,
        device="cpu",
    )
    with pytest.raises(ValueError, match="no legal action"):
        agent.remember(
            [0, 0],
            0,
            0.0,
            [0, 0],
            False,
            [1, 0],
            [0, 0],
        )


def test_no_idle_vm_advances_without_transition(smoke_config):
    adapter = CEWSEnvAdapter(smoke_config, 32)
    adapter.reset()
    task_id = adapter.advance_until_actionable()
    adapter.execute(task_id, adapter.legal_vm_ids(task_id)[0])
    next_finish = float(adapter.env.event_heap[0][0])
    for index in range(1, adapter.num_vms):
        adapter.env.vm_available_at[index] = next_finish + 10.0
    before = adapter.current_time
    next_task = adapter.advance_until_actionable()
    assert next_task is not None
    assert adapter.current_time > before
    # Earlier workflow arrivals may be processed before the first VM finish;
    # no RA transition is produced for any of those all-busy decision points.
    assert adapter.no_legal_vm_advances >= 1
    assert adapter.current_time >= next_finish


def test_four_rule_serialization_and_schema(tmp_path):
    programs = [
        GPProgram((name,), GP_TERMINALS) for name in GP_TERMINALS[:4]
    ]
    path = tmp_path / "rules.json"
    write_json(
        path,
        {
            "schema_version": 2,
            "method_id": "drlea_nichgp",
            "terminal_version": "drlea_gp14_v1",
            "terminal_names": list(GP_TERMINALS),
            "rule_count": 4,
            "rules": [
                {"rule_id": index, **program.to_dict()}
                for index, program in enumerate(programs)
            ],
        },
    )
    loaded = load_rules(path)
    assert len(loaded) == 4
    assert [program.tokens for program in loaded] == [
        program.tokens for program in programs
    ]


def test_nan_inf_rule_is_invalid():
    program = GPProgram(("mul", "READY_COUNT", "READY_WORK"), GP_TERMINALS)
    assert np.isnan(program([np.inf] + [1.0] * 13))
    assert protected_division(1.0, 0.0) == 1.0
    situation = type(
        "Situation",
        (),
        {"terminals": ((np.inf,) + (1.0,) * 13,)},
    )()
    with pytest.raises(ValueError, match="NaN or Inf"):
        behavior_characterization(program, [situation])


def test_three_timelines_risk_and_energy_formula(adapter, smoke_config):
    task_id = adapter.advance_until_actionable()
    prediction = adapter.task_vm_prediction(
        task_id, adapter.legal_vm_ids(task_id)[0]
    )
    assert (
        prediction["optimistic_finish"]
        <= prediction["modal_finish"]
        <= prediction["pessimistic_finish"]
    )
    assert prediction["risk_finish"] == pytest.approx(
        0.05 * prediction["modal_finish"]
        + 0.95 * prediction["pessimistic_finish"]
    )
    adapter.execute(task_id, adapter.legal_vm_ids(task_id)[0])
    energy = adapter.env.get_fuzzy_energy_summary()
    assert energy["fuzzy_total_energy_score"] == pytest.approx(
        energy["fuzzy_total_energy_mean"]
        + energy["fuzzy_total_energy_std"]
    )


def test_lexicographic_violation_first():
    feasible = {
        "deadline_violation_rate": 0.0,
        "max_fuzzy_lateness": 0.0,
        "mean_fuzzy_lateness": 0.0,
        "fuzzy_energy_score": 1000.0,
    }
    low_energy_violation = {
        "deadline_violation_rate": 0.01,
        "max_fuzzy_lateness": 0.1,
        "mean_fuzzy_lateness": 0.01,
        "fuzzy_energy_score": 1.0,
    }
    assert comparison_key(feasible) < comparison_key(low_energy_violation)


def test_checkpoint_round_trip(tmp_path):
    agent = MaskedD3QN(
        3,
        2,
        AgentConfig(
            hidden_dims=(4, 4),
            replay_capacity=8,
            batch_size=1,
        ),
        seed=9,
        device="cpu",
    )
    path = agent.save(tmp_path / "agent.pt")
    restored = MaskedD3QN.load(path, device="cpu")
    assert restored.state_dim == 3
    assert restored.action_dim == 2
    for left, right in zip(
        agent.online.parameters(), restored.online.parameters()
    ):
        assert torch.equal(left, right)


def test_seed_partitions_are_strict():
    with pytest.raises(ValueError, match="strictly disjoint"):
        ensure_disjoint_seeds((1,), (2,), (1,))


def test_comparison_does_not_import_primary_private_agents():
    package = (
        Path(__file__).resolve().parents[1]
        / "algorithms"
        / "comparisons"
        / "drlea_nichgp"
    )
    forbidden = {
        "base.d3qn_agent",
        "algorithms.llm_safe_hrl.base.d3qn_agent",
        "base.safety_shield",
        "base.safety_fallback",
        "base.safety_lagrange",
        "algorithms.llm_safe_hrl.LLM",
    }
    imported = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
    assert imported.isdisjoint(forbidden)


def test_small_complete_three_stage_pipeline():
    config = build_config("SS", "T", 41, smoke=True)
    result = run_pipeline(config)
    output = Path(result["output_dir"])
    assert Path(result["ra_checkpoint"]).is_file()
    assert Path(result["rules_file"]).is_file()
    assert Path(result["sa_checkpoint"]).is_file()
    assert (output / "eval.json").is_file()
    assert (output / "eval.csv").is_file()
    assert (output / "manifest.json").is_file()
    payload = json.loads(
        (output / "rules.json").read_text(encoding="utf-8")
    )
    assert payload["rule_count"] == 4
    assert len(result["evaluation"]["comparison_key"]) == 4
