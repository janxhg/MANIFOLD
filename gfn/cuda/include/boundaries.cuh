#ifndef BOUNDARIES_CUH
#define BOUNDARIES_CUH

#include <cuda_runtime.h>
#include <cmath>

// Topology Constants
#define TOPOLOGY_EUCLIDEAN 0
#define TOPOLOGY_TORUS 1

#define TWO_PI 6.28318530718f
#define PI 3.14159265359f

/**
 * Apply Boundary Condition to Position vector component x.
 * 
 * @param x Component value
 * @param topology Topology ID (0=Euclidean, 1=Torus)
 * @return Adjusted component value
 */
__device__ __forceinline__ float apply_boundary(float x, int topology) {
    if (topology == TOPOLOGY_TORUS) {
        // Map to [0, 2*PI)
        float val = fmodf(x, TWO_PI);
        if (val < 0.0f) val += TWO_PI;
        return val;
    } else {
        // Euclidean
        // Legacy clamp to prevent infinity, but permissive enough for drift
        return fminf(fmaxf(x, -1000.0f), 1000.0f); 
    }
}

/**
 * Calculate distance between two points x1 and x2 on the manifold component-wise.
 * Used for loss functions or relative distance checks.
 * 
 * @param x1 Point 1
 * @param x2 Point 2
 * @param topology Topology ID
 * @return Signed distance (x1 - x2) respecting topology
 */
__device__ __forceinline__ float apply_distance(float x1, float x2, int topology) {
    float dx = x1 - x2;
    if (topology == TOPOLOGY_TORUS) {
        // Smallest distance on circle
        // If dx > PI, go the other way (subtract 2PI)
        // If dx < -PI, add 2PI
        // But for simple "forward" integration we just return raw difference?
        // No, for gradients we need the shortest path vector.
        
        // Wrap dx to [-PI, PI]
        // dx = fmodf(dx + PI, TWO_PI) - PI; // Standard formula
        
        // Optimized branchless or branching? Branching is fine for now.
        if (dx > PI) dx -= TWO_PI;
        else if (dx < -PI) dx += TWO_PI;
    }
    return dx;
}

#endif // BOUNDARIES_CUH
