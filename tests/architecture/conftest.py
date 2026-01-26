import pytest
import torch
import json
import os
import time
from datetime import datetime

class MetricsLogger:
    """
    Professional Logger for Architectural Metrics.
    Saves results in .data/metrics/architecture/ for posterior analysis.
    """
    def __init__(self, test_name):
        self.test_name = test_name
        self.start_time = time.time()
        self.metrics = {
            "test": test_name,
            "timestamp": datetime.now().isoformat(),
            "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU",
            "results": {}
        }
        self.base_path = "D:/ASAS/projects/GFN/.data/metrics/architecture"
        os.makedirs(self.base_path, exist_ok=True)

    def log(self, key, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
        self.metrics["results"][key] = value

    def finish(self):
        self.metrics["duration_seconds"] = time.time() - self.start_time
        path = os.path.join(self.base_path, f"{self.test_name}_{int(time.time())}.json")
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=4)
        print(f"\n[METRICS] Saved {self.test_name} results to {path}")

@pytest.fixture
def metrics(request):
    logger = MetricsLogger(request.node.name)
    yield logger
    logger.finish()

class GeodesicProbe:
    """
    Utility to track particles in manifold space.
    """
    @staticmethod
    def compute_hamiltonian(x, v, metric_fn):
        g = metric_fn(x)
        # Assuming diagonal metric for efficiency in tests
        energy = 0.5 * torch.sum(g * v.pow(2), dim=-1)
        return energy

    @staticmethod
    def estimate_convergence_order(errors, dts):
        """
        Estimates the numerical convergence order p from log-log slope.
        """
        import numpy as np
        log_dts = np.log(dts)
        log_errors = np.log(errors)
        coeffs = np.polyfit(log_dts, log_errors, 1)
        return coeffs[0]

@pytest.fixture
def probe():
    return GeodesicProbe()
