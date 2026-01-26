#pragma once
#include "physics_primitives.cuh"

// ==========================================
// Manifold Topology Logic
// ==========================================

enum TopologyType {
    EUCLIDEAN = 0,
    TORUS = 1,
    SPHERICAL = 2,
    HYPERBOLIC = 3
};

// 1. Manifold Wrappings (The "Pac-Man" effect)
__device__ __forceinline__ float apply_boundary(float x, int topology) {
    if (topology == TORUS) {
        // Wrap to [0, 2pi]
        const float TWO_PI = 2.0f * PI;
        return x - floorf(x / TWO_PI) * TWO_PI;
    }
    return x; // Euclidean is unbounded
}

// 2. Shortest Distance on Manifold
// Used for gradients (x_target - x_pred)
__device__ __forceinline__ float apply_distance(float x1, float x2, int topology) {
    float dx = x1 - x2;
    if (topology == TORUS) {
        const float TWO_PI = 2.0f * PI;
        // Wrap dx to [-PI, PI]
        dx = fmodf(dx + PI, TWO_PI) - PI; 
        if (dx < -PI) dx += TWO_PI; 
    }
    return dx;
}

// 3. Potential Gradient dV/dx dependent on Topology
__device__ __forceinline__ float topology_potential_grad(float x, float V_w, int topology) {
    if (topology == TORUS) {
        return cosf(x) * V_w;
    } else {
        return V_w; // Linear for Euclidean/Hyperbolic local patch
    }
}

// 4. Metric Embedding (Coordinate --> Feature)
__device__ __forceinline__ void compute_metric_embedding(
    float* __restrict__ embedding_out, 
    const float x_val,
    int topology,
    int dim_idx,
    int total_dim
) {
    if (topology == TORUS) {
        embedding_out[0] = sinf(x_val);
        embedding_out[1] = cosf(x_val);
    } else {
        embedding_out[0] = x_val;
        embedding_out[1] = 0.0f; 
    }
}
