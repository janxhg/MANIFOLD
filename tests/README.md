# MANIFOLD Professional Test Suite

This directory contains the verification suite for MANIFOLD (formerly GFN). All tests are unified and consistent with the V2 Architecture (Multi-Head Geodesic Flows).

## Structure

### 1. 🏗️ `unit/` (Component Logic)
Tests individual classes in isolation.
*   `test_components.py`: Verifies `Manifold`, `MLayer`, and `RiemannianGating` shapes and forward passes.

### 2. ⚛️ `physics/` (Mathematical Correctness)
Ensures the model adheres to physical laws.
*   `test_mechanics.py`: Checks Geodesic equations, energy conservation (Symplectic integrity), and gradient flows. **Critical for "Physics-Informed" claims.**

### 3. 🔌 `integration/` (Training Loops)
Tests the full training pipeline.
*   `test_overfit_sanity.py`: Can the model overfit a single batch? (Smoke test).
*   `test_full_training.py`: Runs a complete training epoch on dummy data.

### 4. 📊 `benchmarks/` (Performance & Metrics)
Quantitative analysis of model capabilities.
*   `benchmark_performance.py`: Measures Tokens/sec throughput compared to Transformers.
*   `benchmark_copy_task.py`: Evaluates memory capacity (Copy/Paste).
*   `benchmark_math_task.py`: Evaluates reasoning/compositional generalization.
*   `benchmark_ood.py`: Tests Out-of-Distribution length generalization.

## Usage

Run all tests:
```bash
python tests/run_suite.py
```

Run specific benchmark:
```bash
python tests/benchmarks/benchmark_performance.py
```
