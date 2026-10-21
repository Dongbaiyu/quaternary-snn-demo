from __future__ import annotations

import gc
import tracemalloc
from unittest.mock import patch

import pytest
import torch

from inference.demo_samples import load_demo_samples
from inference.predictor import Predictor
from qsn_model.constants import Config, PROJECT_ROOT
from qsn_model.neurons import QuaternaryLIF


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    instance = Predictor(Config())
    instance.load_checkpoint(
        str(PROJECT_ROOT / "checkpoints" / "qsn_sresnet20_t4_best.pth")
    )
    return instance


def hook_count(model) -> int:
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks)
        for module in model.modules()
    )


def test_reset_between_runs_and_count_values(predictor):
    first = predictor.predict_cifar_index(143)
    second = predictor.predict_cifar_index(143)

    assert first.predicted_class == second.predicted_class == "cat"
    assert first.spike_statistics == second.spike_statistics
    assert set(first.spike_statistics) == {-1, 0, 1, 2}
    assert all(isinstance(count, int) and count >= 0 for count in first.spike_statistics.values())

    total = sum(first.spike_statistics.values())
    assert total > 0
    zero_rate = first.spike_statistics[0] / total
    nonzero_rate = 1.0 - zero_rate
    assert zero_rate + nonzero_rate == pytest.approx(1.0)


def test_dynamic_time_steps_reset_and_recount_without_hooks(predictor):
    model = predictor.model
    assert model is not None
    before_hooks = hook_count(model)
    totals = {}

    for time_steps in (1, 2, 4, 6):
        with (
            patch.object(model, "forward", wraps=model.forward) as model_forward,
            patch.object(model.lif1, "forward", wraps=model.lif1.forward) as input_lif,
        ):
            result = predictor.predict_cifar_index(
                143, time_steps=f"T = {time_steps}"
            )
        assert model_forward.call_count == 1
        assert model_forward.call_args.kwargs["time_steps"] == time_steps
        assert input_lif.call_count == time_steps
        assert result.time_steps == time_steps
        assert len(result.top3_classes) == 3
        assert sum(result.class_scores.values()) == pytest.approx(1.0)
        assert set(result.spike_statistics) == {-1, 0, 1, 2}
        assert all(count >= 0 for count in result.spike_statistics.values())
        totals[time_steps] = sum(result.spike_statistics.values())
        assert hook_count(model) == before_hooks

    assert totals[1] > 0
    for time_steps in (1, 2, 4, 6):
        assert totals[time_steps] == totals[1] * time_steps
    assert hook_count(model) == before_hooks == 0


def test_statistics_do_not_change_logits_or_add_hooks(predictor):
    model = predictor.model
    assert model is not None
    dataset = predictor._testset
    image, _ = dataset[143]
    inputs = image.unsqueeze(0).to(predictor.device)

    before_hooks = hook_count(model)
    model.reset_spike_statistics()
    with torch.no_grad():
        first_logits = model(inputs)
    first_counts = model.spike_statistics()

    model.reset_spike_statistics()
    with torch.no_grad():
        second_logits = model(inputs)
    second_counts = model.spike_statistics()

    torch.testing.assert_close(second_logits, first_logits, rtol=0.0, atol=0.0)
    assert first_counts == second_counts
    assert hook_count(model) == before_hooks == 0

    module_total = sum(
        sum(module.inference_spike_counts.values())
        for module in model.modules()
        if isinstance(module, QuaternaryLIF)
    )
    assert sum(second_counts.values()) == module_total


def test_twenty_inferences_do_not_accumulate_hooks_or_memory(predictor):
    model = predictor.model
    assert model is not None
    indices = [sample.index for sample in load_demo_samples()]
    assert indices

    predictor.predict_cifar_index(indices[0])
    before_hooks = hook_count(model)
    gc.collect()

    if predictor.device.type == "cuda":
        torch.cuda.synchronize(predictor.device)
        baseline = torch.cuda.memory_allocated(predictor.device)
        for run in range(20):
            predictor.predict_cifar_index(indices[run % len(indices)])
        torch.cuda.synchronize(predictor.device)
        gc.collect()
        growth = torch.cuda.memory_allocated(predictor.device) - baseline
        assert growth < 32 * 1024 * 1024
    else:
        tracemalloc.start()
        baseline, _ = tracemalloc.get_traced_memory()
        for run in range(20):
            predictor.predict_cifar_index(indices[run % len(indices)])
        gc.collect()
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert current - baseline < 16 * 1024 * 1024

    assert hook_count(model) == before_hooks == 0
