# Future Integration: ternary-reservoir

## Current State
Provides echo state networks on {-1, 0, +1} with sparse random reservoir matrices, reservoir dynamics, ridge regression readout, and spectral radius control for temporal pattern recognition.

## Integration Opportunities

### With ternary-cell (Temporal Pattern Recognition)
Cell grids produce time series of states. `ternary-reservoir` recognizes patterns in those time series. Train a reservoir on historical cell grid states; it predicts future states. When the actual state diverges from the prediction, that's a "surprise" — feeding directly back into the cell's surprise computation. The reservoir becomes the cell's temporal memory.

### With ternary-music (Rhythm Recognition)
Musical patterns are temporal. A reservoir trained on ternary rhythmic sequences recognizes when a room is "in rhythm" vs. "arrhythmic." This is the temporal analog of spatial pattern recognition.

### With ternary-chaos
Reservoir computers are excellent at predicting chaotic systems. Train a reservoir on the iterated map output from `ternary-chaos`; it can predict short-term chaotic dynamics. The prediction horizon (how far ahead the reservoir can predict) measures the system's Lyapunov exponent — connecting back to chaos theory.

## Potential in Mature Systems
In room-as-codespace, each room generates a time series of states. Reservoir computing predicts room behavior: which rooms will need more resources, which are heading toward failure, which are settling into stable patterns. The spectral radius parameter controls how much temporal history the reservoir retains — longer history for slow-changing rooms, shorter for volatile rooms.

## Cross-Pollination Ideas
- Reservoir as a universal temporal encoder — compress arbitrary-length room histories into fixed-size state vectors
- Ridge regression readout as a lightweight prediction model for room resource needs
- Spectral radius as a tunable "memory horizon" per room

## Dependencies for Next Steps
- ternary-cell needs temporal state history for reservoir input
- Integration with ternary-chaos for chaotic system prediction
- Performance profiling: reservoir training cost vs. prediction benefit
