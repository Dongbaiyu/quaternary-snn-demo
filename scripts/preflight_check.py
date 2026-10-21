from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_IMPORTS = (
    "torch",
    "torchvision",
    "PySide6",
    "numpy",
    "pandas",
    "matplotlib",
)
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "qsn_sresnet20_t4_best.pth"
METADATA = CHECKPOINT.with_suffix(".json")
TRAINING_CURVE = PROJECT_ROOT / "results" / "quaternary_result_v10.png"
CONFUSION_MATRIX = PROJECT_ROOT / "results" / "confusion_matrix_v10.png"
DEMO_SAMPLES = PROJECT_ROOT / "configs" / "demo_samples.json"


def version_of(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "installed"))


def find_dataset() -> Path | None:
    candidates = (
        PROJECT_ROOT / "data",
        PROJECT_ROOT.parent / "Demo" / "data",
    )
    for root in candidates:
        if (root / "cifar-10-batches-py" / "test_batch").is_file():
            return root
    return None


def validate_metadata() -> tuple[bool, str]:
    if not METADATA.is_file():
        return False, f"MISSING ({METADATA})"
    try:
        payload = json.loads(METADATA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"INVALID ({exc})"
    expected = {
        "model": "SResNet20",
        "spike_type": "quaternary",
        "spike_values": [-1, 0, 1, 2],
        "time_steps": 4,
        "tau": 0.875,
        "dataset": "CIFAR-10",
        "test_accuracy": 92.03,
        "quantization": "FP32",
        "source": "qua_v10.py",
    }
    mismatches = [
        key for key, value in expected.items() if payload.get(key) != value
    ]
    if mismatches:
        return False, f"INVALID fields: {', '.join(mismatches)}"
    return True, f"PASS ({METADATA.name}, T=4, 92.03%)"


def validate_demo_samples() -> tuple[bool, str]:
    if not DEMO_SAMPLES.is_file():
        return False, f"MISSING ({DEMO_SAMPLES})"
    try:
        payload = json.loads(DEMO_SAMPLES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"INVALID ({exc})"
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or len(samples) < 5:
        return False, "INVALID (requires at least five samples)"
    labels = {
        item.get("true_label")
        for item in samples
        if isinstance(item, dict) and item.get("verified") is True
    }
    valid = all(
        isinstance(item, dict)
        and isinstance(item.get("index"), int)
        and 0 <= item["index"] < 10000
        and item.get("verified") is True
        and item.get("true_label") == item.get("expected_prediction")
        for item in samples
    )
    if not valid or len(labels) < 5:
        return False, "INVALID (unverified, incorrect, or insufficient class coverage)"
    return True, f"PASS ({len(samples)} verified samples, {len(labels)} classes)"


def run_parity() -> tuple[bool, str]:
    command = [sys.executable, "-m", "pytest", "tests/test_parity.py", "-q"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    lines = [
        line.strip()
        for line in (completed.stdout + completed.stderr).splitlines()
        if line.strip()
    ]
    summary = next((line for line in reversed(lines) if "passed" in line), "")
    return completed.returncode == 0, (
        f"PASS ({summary})" if completed.returncode == 0 else f"FAIL ({lines[-1] if lines else 'no output'})"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    package_versions = {}
    imports_ok = True
    for name in REQUIRED_IMPORTS:
        try:
            package_versions[name] = version_of(name)
        except Exception as exc:
            imports_ok = False
            package_versions[name] = f"MISSING ({exc})"

    torch_module = None
    try:
        torch_module = importlib.import_module("torch")
        torch_text = str(torch_module.__version__)
        cuda_available = bool(torch_module.cuda.is_available())
        cuda_device = (
            torch_module.cuda.get_device_name(torch_module.cuda.current_device())
            if cuda_available
            else "N/A"
        )
    except Exception as exc:
        torch_text = f"MISSING ({exc})"
        cuda_available = False
        cuda_device = "N/A"

    dataset = find_dataset()
    checkpoint_ok = CHECKPOINT.is_file()
    metadata_ok, metadata_text = validate_metadata()
    curve_ok = TRAINING_CURVE.is_file()
    matrix_ok = CONFUSION_MATRIX.is_file()
    samples_ok, samples_text = validate_demo_samples()
    parity_ok, parity_text = run_parity() if imports_ok and checkpoint_ok else (
        False,
        "SKIPPED (environment or checkpoint missing)",
    )

    print(
        "Python environment:",
        "PASS" if imports_ok else "FAIL",
        f"({sys.version.split()[0]}; "
        + ", ".join(f"{name}={version}" for name, version in package_versions.items())
        + ")",
    )
    print("Torch:", torch_text)
    print("CUDA available:", cuda_available)
    print("CUDA device:", cuda_device)
    print("PySide6:", package_versions["PySide6"])
    print("Dataset:", f"PASS ({dataset})" if dataset else "MISSING")
    print(
        "Checkpoint:",
        f"PASS ({CHECKPOINT.name}, {CHECKPOINT.stat().st_size} bytes, SHA-256 {sha256(CHECKPOINT)})"
        if checkpoint_ok
        else f"MISSING ({CHECKPOINT})",
    )
    print("Metadata:", metadata_text)
    print(
        "Training curve:",
        f"PASS ({TRAINING_CURVE})" if curve_ok else f"MISSING ({TRAINING_CURVE})",
    )
    print(
        "Confusion matrix:",
        f"PASS ({CONFUSION_MATRIX})" if matrix_ok else f"MISSING ({CONFUSION_MATRIX})",
    )
    print("Demo samples:", samples_text)
    print("Parity status:", parity_text)

    ready = all(
        (
            imports_ok,
            dataset is not None,
            checkpoint_ok,
            metadata_ok,
            curve_ok,
            matrix_ok,
            samples_ok,
            parity_ok,
        )
    )
    print("Demo readiness:", "READY" if ready else "NOT READY")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
