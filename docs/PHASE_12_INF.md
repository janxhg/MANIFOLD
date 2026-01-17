# Phase 12: Implicit Neural Fields (INFs)

**Status:** Implemented & Verified (v3.0)
**Date:** 2026-01-17

## 1. Concept
We transitioned from **Discrete Embeddings** (`nn.Embedding` table) to **Implicit Continuous Functions** (INFs).
Instead of looking up a static vector, the model:
1.  Looks up a **Low-Rank Coordinate** $c \in \mathbb{R}^{16}$.
2.  Passes it through a **SIREN (Sine ResNet)** MLP.
3.  Outputs the high-dimensional embedding vector.

$$ E(token) = \Phi_{\theta}(c_{token}) $$

### 2. Implementation
### `src/embeddings.py`
Two modes available:
1.  **`ImplicitEmbedding` (Hybrid)**: Learnable Coordinate Table `[Vocab, 16]` -> SIREN.
    *   Good for: Fixed vocabularies where we want topology but some learnability.
2.  **`FunctionalEmbedding` (Pure)**: Procedural Coordinate (Hash) -> SIREN.
    *   **No Param Table**. Coordinates are math functions of the ID.
    *   Good for: True "Infinite Vocabulary" inputs.

### 3. Benefits
*   **Topology:** Tokens now live in a continuous metric space.
*   **Memory Efficiency:**
    *   Standard (1M): **513M Params** (6.8 GB VRAM)
    *   Functional (1M): **257M Params** (5.8 GB VRAM)
    *   Infinite (1M): **0.31M Params** (**30 MB VRAM**)
    *   **Conclusion:** The "Infinite" mode (Functional Input + Implicit Readout) achieves **True O(1) Memory Scaling**. It uses ~30MB VRAM regardless of vocabulary size (up to 1M+ tested).

## 4. Verification
Ran `tests/benchmarks/benchmark_inf_vram.py`.
*   **Input Scaling:** Functional Embedding completely removes input layer impact.
*   **Output Scaling:** Implicit Readout completely removes output layer impact.
*   **Result:** A model that can conceptually handle **Infinite Vocabularies** with constant parameter cost.

## 5. Usage
Set in `config.yaml`:
```yaml
physics:
  embedding:
    type: functional 
    coord_dim: 16
  readout:
    type: implicit
```
