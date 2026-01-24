# Plan de Sincronización: Loss vs Accuracy (Toroide)

Este documento detalla las acciones necesarias para sincronizar el Loss con el Accuracy en el benchmark toroidal, eliminando la "fricción numérica" que impide la convergencia total.

## 1. Ajustes en MLayer (gfn/layers/base.py)
- **Bypass de Normalización**: Desactivar `RMSNorm` para el estado $x$ cuando `topology == torus`. La normalización lineal corrompe la periodicidad angular.
- **Reducción de Fricción Inicial**: Cambiar el bias inicial de `friction_gates` de `3.0` (freno total) a `0.0` o `-1.0` para permitir que el modelo fluya libremente hacia el target en las primeras épocas.

## 2. Ajustes en el Benchmark (tests/benchmarks/viz/vis_gfn_superiority.py)
- **Aumento de Potencia**: Incrementar `impulse_scale` de `5.0` a `10.0`. Esto asegura que un solo impulso sea suficiente para mover el estado una distancia de $\pi$ (salto de paridad).
- **Métrica Estricta**: Reducir el umbral de Accuracy de $\pi/2$ a $0.5$ radianes. Esto forzará al modelo a no conformarse con estar en el "hemisferio correcto" y buscar el centro del target.

## 3. Verificación de Integradores
- Asegurar que la lógica de `apply_boundary` esté activa en todos los integradores de `gfn/integrators/symplectic/` para evitar que las coordenadas $x$ se escapen del rango $[0, 2\pi)$.
