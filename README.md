# ternary-reservoir

Reservoir computing with ternary nodes — echo state networks that quantize internal state to `{-1, 0, +1}` with sparse reservoir matrices, spectral radius control, and ridge regression readout.

## Why This Exists

Echo State Networks (ESNs) are powerful for time-series prediction and classification, but their internal reservoir states are continuous-valued floating-point vectors. In hardware-constrained scenarios (neuromorphic chips, FPGA, edge AI), maintaining continuous state is expensive. By quantizing reservoir activations to three values while preserving the dynamics through spectral radius control and leaky integration, this crate enables reservoir computing in environments where floating-point state is impractical.

## Core Concepts

- **Ternary** — Enum type: `Neg` (−1), `Zero` (0), `Pos` (+1)
- **ReservoirMatrix** — Sparse random matrix with configurable density and spectral radius
- **TernaryReservoir** — Recurrent reservoir with ternary-quantized state and leaky integration
- **RidgeReadout** — Linear readout trained via ridge regression (closed-form solution)
- **EchoStateNetwork** — Complete ESN combining reservoir + readout

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-reservoir = "0.1"
```

```rust
use ternary_reservoir::*;

// Create an echo state network
let mut esn = EchoStateNetwork::new(
    50,    // reservoir size
    2,     // input dimensions
    1,     // output dimensions
    0.9,   // spectral radius
    0.3,   // leak rate
    0.01,  // ridge parameter
    42,    // random seed
);

// Training data: predict the next value in a sequence
let inputs: Vec<Vec<f64>> = (0..100)
    .map(|t| vec![(t as f64 * 0.1).sin(), (t as f64 * 0.1).cos()])
    .collect();
let targets: Vec<Vec<f64>> = inputs.iter().skip(1)
    .map(|v| vec![v[0]])
    .chain(std::iter::once(vec![0.0]))
    .collect();

// Train with 10-step washout
esn.train(&inputs, &targets, 10);

// Predict
let output = esn.predict(&[0.5, 0.866]);
println!("Prediction: {:.4}", output[0]);

// Batch prediction
let predictions = esn.predict_sequence(&inputs[..10]);
for (i, pred) in predictions.iter().enumerate() {
    println!("Step {}: {:.4}", i, pred[0]);
}

// Mean squared error
let mse = mse(&predictions, &targets[..10].to_vec());
println!("MSE: {:.6}", mse);
```

## API Overview

| Type | Description |
|---|---|
| `ReservoirMatrix` | Sparse matrix with `mul_vec`, `spectral_radius`, `scale_spectral_radius` |
| `TernaryReservoir` | Reservoir with `update`, `run`, `washout`, `reset`, `state_f64` |
| `RidgeReadout` | Readout with `train` (normal equation) and `predict` |
| `EchoStateNetwork` | Full ESN with `train`, `predict`, `predict_sequence` |
| `mse` | Mean squared error between predictions and targets |

## How It Works

1. **Sparse Reservoir**: A random sparse matrix (default 30% density) with ±1 entries is generated, then scaled so its largest eigenvalue (spectral radius) matches the target. A spectral radius < 1.0 ensures the echo state property (dynamics fade over time).

2. **Ternary Quantization**: At each timestep, the activation is computed as `leak × prev_state + (1 − leak) × (W_res × state + W_in × input)`. The result is quantized: above 0.25 → +1, below −0.25 → −1, else 0.

3. **Washout**: Initial timesteps are discarded to let transient dynamics settle before collecting states for training.

4. **Ridge Regression**: The readout weights are computed in closed form via the normal equation `W = YXᵀ(XXᵀ + λI)⁻¹`, where λ prevents overfitting on correlated features.

## Use Cases

1. **Edge time-series forecasting** — Deploy lightweight reservoir predictors on microcontrollers for sensor prediction
2. **Neuromorphic computing** — Map ternary reservoirs to memristor crossbar arrays or FPGA look-up tables
3. **Speech/audio classification** — Use ternary reservoirs as feature extractors for lightweight audio classifiers
4. **Robotics control** — Low-latency control policies with ternary-quantized recurrent dynamics

## Ecosystem

Part of the **SuperInstance** ternary computing crate family:

- `ternary-compression-v2` — Multi-algorithm ternary compression
- `ternary-hash` — Hashing and fingerprinting for ternary data
- `ternary-pca` — Principal component analysis on ternary values
- `ternary-ga` — Genetic algorithms with ternary genomes
- `ternary-matrix` — Compact ternary matrix operations
- `ternary-evolution-advanced` — Advanced evolutionary optimization
- `ternary-geometry` — Geometric algorithms in ternary space
- `ternary-causality` — Causal inference for ternary systems
- `ternary-consensus` — Distributed consensus for ternary agents

## License

MIT
