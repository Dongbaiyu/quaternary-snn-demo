from __future__ import annotations

import json

import pytest
import torch

from inference.model_registry import MODEL_REGISTRY, get_model_spec, sha256_file
from inference.predictor import Predictor
from qsn_model.constants import Config
from qsn_model.neurons import QuaternaryLIF
from web_app.components import create_cpu_service, load_sample_image, load_web_samples


EXPECTED = {
    1: (0.875, "quaternary_sresnet20_qua_t_1_best_weights.pth", "qua_timestep_1.py", 90.99, 9099),
    2: (0.875, "quaternary_sresnet20_qua_t_2_best_weights.pth", "qua_timestep_2.py", 91.93, 9193),
    4: (0.875, "qsn_sresnet20_t4_best.pth", "qua_v10.py", 92.03, 9203),
    6: (0.9, "quaternary_sresnet20_qua_t_6_best_weights.pth", "qua_timestep_6.py", 92.07, 9207),
}


@pytest.mark.parametrize("time_steps", (1, 2, 4, 6))
def test_registry_assets_and_verified_metadata(time_steps):
    tau, filename, source, accuracy, correct = EXPECTED[time_steps]
    spec = get_model_spec(time_steps)
    assert spec is MODEL_REGISTRY[time_steps]
    assert spec.time_steps == time_steps
    assert spec.tau == tau
    assert spec.checkpoint.name == filename
    assert spec.source == source
    assert spec.checkpoint.is_file()
    assert spec.metadata_path.is_file()
    assert spec.confusion_matrix.is_file()
    assert spec.training_result.is_file()
    assert spec.recorded_train_accuracy == {1: 98.96, 2: 99.57, 4: 99.84, 6: 99.84}[time_steps]
    assert spec.recorded_test_accuracy == {1: 91.18, 2: 92.04, 4: 92.03, 6: 92.30}[time_steps]
    metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
    assert metadata["time_steps"] == time_steps
    assert metadata["tau"] == tau
    assert metadata["source"] == source
    assert metadata["test_accuracy"] == accuracy
    assert metadata["correct_count"] == correct
    assert metadata["test_count"] == 10000
    assert metadata["checkpoint_sha256"] == sha256_file(spec.checkpoint)


@pytest.mark.parametrize("time_steps", (1, 2, 4, 6))
def test_each_checkpoint_loads_strictly_and_all_lif_layers_use_registry_tau(time_steps):
    spec = get_model_spec(time_steps)
    service = create_cpu_service(spec.checkpoint)
    predictor = service.predictor
    assert predictor.checkpoint is not None
    assert predictor.checkpoint.path == spec.checkpoint.resolve()
    model = predictor.model
    assert model is not None
    assert model.time_steps == time_steps
    assert model.tau == spec.tau
    lif_layers = [module for module in model.modules() if isinstance(module, QuaternaryLIF)]
    assert len(lif_layers) == 19
    assert all(module.decay == spec.tau for module in lif_layers)


@pytest.mark.parametrize("time_steps", (1, 2, 4, 6))
def test_registered_predictor_executes_exact_time_step_loop(time_steps):
    spec = get_model_spec(time_steps)
    service = create_cpu_service(spec.checkpoint)
    model = service.predictor.model
    assert model is not None
    calls = 0

    def count_input_lif(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.lif1.register_forward_hook(count_input_lif)
    try:
        sample = load_web_samples()[0]
        result = service.predict_image(load_sample_image(sample), time_steps=time_steps)
    finally:
        handle.remove()
    assert calls == time_steps
    assert result.time_steps == time_steps
    assert result.training_time_steps == time_steps
    assert result.tau == spec.tau
    assert result.checkpoint_name == spec.checkpoint.name
    assert result.checkpoint_sha256 == sha256_file(spec.checkpoint)
    assert not model.lif1._forward_hooks


def test_predictor_instances_and_models_do_not_cross_registry_cache_identity():
    services = {time_steps: create_cpu_service(spec.checkpoint) for time_steps, spec in MODEL_REGISTRY.items()}
    assert len({id(service.predictor) for service in services.values()}) == 4
    assert len({id(service.predictor.model) for service in services.values()}) == 4
    assert services[6].predictor.model.tau == 0.9
    assert services[4].predictor.model.tau == 0.875


def test_all_checkpoint_state_dicts_are_architecture_compatible_strict_true():
    for time_steps, spec in MODEL_REGISTRY.items():
        predictor = Predictor(Config(device="cpu", mode="inference"))
        predictor.load_checkpoint(str(spec.checkpoint))
        assert predictor.model is not None
        assert predictor.checkpoint.metadata["time_steps"] == time_steps
