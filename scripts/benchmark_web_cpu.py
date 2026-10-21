from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

PROCESS_STARTED = time.perf_counter()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
import torch

from web_app.components import (
    CHECKPOINT_PATH,
    create_cpu_service,
    load_sample_image,
    load_web_samples,
)
def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main() -> int:
    before_model = rss_mb()
    load_started = time.perf_counter()
    service = create_cpu_service(CHECKPOINT_PATH)
    model_load_seconds = time.perf_counter() - load_started
    model_ready_elapsed = time.perf_counter() - PROCESS_STARTED
    ready_rss = rss_mb()

    sample = load_web_samples()[0]
    image = load_sample_image(sample)
    first_wall_started = time.perf_counter()
    first = service.predict_image(image)
    first_wall = time.perf_counter() - first_wall_started
    peak_rss = rss_mb()

    warm_wall = []
    warm_model = []
    for _ in range(5):
        wall_started = time.perf_counter()
        result = service.predict_image(image)
        warm_wall.append(time.perf_counter() - wall_started)
        warm_model.append(result.inference_time)
        peak_rss = max(peak_rss, rss_mb())

    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": service.device_label,
        "checkpoint": CHECKPOINT_PATH.name,
        "process_start_to_model_ready_seconds": round(model_ready_elapsed, 6),
        "model_load_seconds": round(model_load_seconds, 6),
        "first_request_wall_seconds": round(first_wall, 6),
        "first_model_inference_seconds": round(first.inference_time, 6),
        "warm_request_wall_mean_seconds": round(statistics.mean(warm_wall), 6),
        "warm_model_inference_mean_seconds": round(statistics.mean(warm_model), 6),
        "rss_before_model_mb": round(before_model, 2),
        "rss_model_ready_mb": round(ready_rss, 2),
        "rss_peak_mb": round(peak_rss, 2),
        "prediction": first.predicted_class,
        "expected_prediction": sample.expected_prediction,
    }
    if payload["prediction"] != payload["expected_prediction"]:
        raise RuntimeError("verified demo prediction changed during benchmark")
    output = PROJECT_ROOT / "results" / "web_cpu_benchmark.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
