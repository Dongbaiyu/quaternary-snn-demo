from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.predictor import scan_checkpoints
from qsn_model.model import SResNet20


def load_original_module():
    path = ROOT / "original" / "qua_v10.py"
    spec = importlib.util.spec_from_file_location("qua_v10_original", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_refactor_matches_original_checkpoint_inference():
    torch.manual_seed(2026)
    original = load_original_module()
    original.TIME_STEPS = 4
    checkpoint = ROOT / "checkpoints" / "qsn_sresnet20_t4_best.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    baseline_model = original.SResNet20()
    refactored_model = SResNet20(time_steps=4, tau=0.875)
    baseline_model.load_state_dict(state, strict=True)
    refactored_model.load_state_dict(state, strict=True)
    baseline_model.eval()
    refactored_model.eval()
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = baseline_model(inputs)
        actual_default = refactored_model(inputs)
        actual_explicit = refactored_model(inputs, time_steps=4)
    torch.testing.assert_close(actual_default, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_explicit, expected, rtol=0.0, atol=0.0)


def test_only_valid_metadata_is_exposed():
    checkpoints = scan_checkpoints(ROOT / "checkpoints")
    assert [(item.metadata["time_steps"], item.path.name) for item in checkpoints] == [
        (4, "qsn_sresnet20_t4_best.pth")
    ]
