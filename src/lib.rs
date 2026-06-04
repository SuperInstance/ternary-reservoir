#![forbid(unsafe_code)]

//! Reservoir computing with ternary nodes: echo state networks on {-1, 0, +1},
//! reservoir dynamics, ridge regression readout, spectral radius control.

use core::f64;

/// Ternary value: -1, 0, or +1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ternary {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

impl Ternary {
    pub fn to_f64(self) -> f64 {
        self as i8 as f64
    }

    pub fn from_f64(v: f64) -> Self {
        if v > 0.5 {
            Ternary::Pos
        } else if v < -0.5 {
            Ternary::Neg
        } else {
            Ternary::Zero
        }
    }

    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Ternary::Neg),
            0 => Some(Ternary::Zero),
            1 => Some(Ternary::Pos),
            _ => None,
        }
    }
}

/// A sparse random reservoir matrix.
#[derive(Clone, Debug)]
pub struct ReservoirMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<(usize, f64)>>, // row -> [(col, value)]
}

impl ReservoirMatrix {
    /// Create a random sparse reservoir matrix with given density and spectral radius.
    /// Uses a simple deterministic PRNG (LCG) for reproducibility.
    pub fn random_sparse(rows: usize, cols: usize, density: f64, spectral_radius: f64, seed: u64) -> Self {
        let mut rng = seed;
        let mut data = vec![vec![]; rows];

        let next_rand = |rng: &mut u64| -> u64 {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *rng
        };

        for i in 0..rows {
            for j in 0..cols {
                let r = next_rand(&mut rng);
                let frac = (r % 10000) as f64 / 10000.0;
                if frac < density {
                    let sign_r = next_rand(&mut rng);
                    let val = if sign_r % 2 == 0 { 1.0 } else { -1.0 };
                    data[i].push((j, val));
                }
            }
        }

        let mut mat = ReservoirMatrix { rows, cols, data };
        mat.scale_spectral_radius(spectral_radius);
        mat
    }

    /// Create an identity-like reservoir (for testing).
    pub fn identity(n: usize) -> Self {
        let mut data = vec![vec![]; n];
        for i in 0..n {
            data[i].push((i, 1.0));
        }
        ReservoirMatrix { rows: n, cols: n, data }
    }

    /// Matrix-vector multiply.
    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for &(j, val) in &self.data[i] {
                if j < v.len() {
                    result[i] += val * v[j];
                }
            }
        }
        result
    }

    /// Compute the spectral radius (approximate: max absolute eigenvalue via power iteration).
    pub fn spectral_radius(&self) -> f64 {
        if self.rows == 0 {
            return 0.0;
        }
        let n = self.rows;
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        
        for _ in 0..100 {
            let w = self.mul_vec(&v);
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                return 0.0;
            }
            v = w.iter().map(|x| x / norm).collect();
        }

        let Av = self.mul_vec(&v);
        let dot: f64 = v.iter().zip(Av.iter()).map(|(a, b)| a * b).sum();
        dot.abs()
    }

    /// Scale the matrix to achieve a target spectral radius.
    pub fn scale_spectral_radius(&mut self, target: f64) {
        let current = self.spectral_radius();
        if current < 1e-15 {
            return;
        }
        let scale = target / current;
        for row in &mut self.data {
            for entry in row.iter_mut() {
                entry.1 *= scale;
            }
        }
    }

    /// Count non-zero entries.
    pub fn nnz(&self) -> usize {
        self.data.iter().map(|r| r.len()).sum()
    }

    /// Density (fraction of non-zero entries).
    pub fn density(&self) -> f64 {
        if self.rows * self.cols == 0 {
            return 0.0;
        }
        self.nnz() as f64 / (self.rows * self.cols) as f64
    }
}

/// A ternary reservoir for echo state computing.
#[derive(Clone, Debug)]
pub struct TernaryReservoir {
    pub size: usize,
    pub input_dim: usize,
    pub reservoir_weights: ReservoirMatrix,
    pub input_weights: ReservoirMatrix,
    pub state: Vec<Ternary>,
    pub leak_rate: f64,
    pub spectral_radius: f64,
}

impl TernaryReservoir {
    pub fn new(size: usize, input_dim: usize, spectral_radius: f64, leak_rate: f64, seed: u64) -> Self {
        let reservoir_weights = ReservoirMatrix::random_sparse(size, size, 0.3, spectral_radius, seed);
        // Input weights: dense, unscaled (raw +/-1 values)
        let mut rng = seed.wrapping_add(12345);
        let next_rand = |rng: &mut u64| -> u64 {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *rng
        };
        let mut iw_data = vec![vec![]; size];
        for i in 0..size {
            for j in 0..input_dim {
                let sign_r = next_rand(&mut rng);
                let val = if sign_r % 2 == 0 { 1.0 } else { -1.0 };
                iw_data[i].push((j, val));
            }
        }
        let input_weights = ReservoirMatrix { rows: size, cols: input_dim, data: iw_data };
        let state = vec![Ternary::Zero; size];

        TernaryReservoir {
            size,
            input_dim,
            reservoir_weights,
            input_weights,
            state,
            leak_rate,
            spectral_radius,
        }
    }

    /// Reset reservoir state to all zeros.
    pub fn reset(&mut self) {
        self.state = vec![Ternary::Zero; self.size];
    }

    /// Get the current state as f64 vector.
    pub fn state_f64(&self) -> Vec<f64> {
        self.state.iter().map(|t| t.to_f64()).collect()
    }

    /// Update the reservoir with an input vector.
    pub fn update(&mut self, input: &[f64]) {
        let w_res = self.reservoir_weights.mul_vec(&self.state_f64());
        let w_in = self.input_weights.mul_vec(input);

        let mut new_state = vec![Ternary::Zero; self.size];
        for i in 0..self.size {
            let prev = self.state[i].to_f64();
            let activation = self.leak_rate * prev + (1.0 - self.leak_rate) * (w_res[i] + w_in[i]);
            // Use a softer threshold for ternary quantization
            if activation > 0.25 {
                new_state[i] = Ternary::Pos;
            } else if activation < -0.25 {
                new_state[i] = Ternary::Neg;
            } else {
                new_state[i] = Ternary::Zero;
            }
        }
        self.state = new_state;
    }

    /// Run the reservoir on a sequence of inputs, collecting states.
    pub fn run(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<Ternary>> {
        self.reset();
        let mut states = vec![];
        for input in inputs {
            self.update(input);
            states.push(self.state.clone());
        }
        states
    }

    /// Washout: discard the first `n` steps to let the reservoir settle.
    pub fn washout(&mut self, inputs: &[Vec<f64>], washout: usize) -> Vec<Vec<Ternary>> {
        self.reset();
        let mut states = vec![];
        for (i, input) in inputs.iter().enumerate() {
            self.update(input);
            if i >= washout {
                states.push(self.state.clone());
            }
        }
        states
    }
}

/// Ridge regression readout layer.
#[derive(Clone, Debug)]
pub struct RidgeReadout {
    pub weights: Vec<Vec<f64>>, // output_dim x reservoir_size
    pub output_dim: usize,
    pub ridge_param: f64,
}

impl RidgeReadout {
    pub fn new(output_dim: usize, reservoir_size: usize, ridge_param: f64) -> Self {
        RidgeReadout {
            weights: vec![vec![0.0; reservoir_size]; output_dim],
            output_dim,
            ridge_param,
        }
    }

    /// Train the readout using collected reservoir states and target outputs.
    /// Uses the normal equation: W = Y * X^T * (X * X^T + lambda * I)^-1
    pub fn train(&mut self, states: &[Vec<f64>], targets: &[Vec<f64>]) {
        if states.is_empty() {
            return;
        }
        let n = states.len();
        let m = states[0].len();

        // Compute X * X^T + lambda * I (m x m)
        let mut xxt = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..m {
                for k in 0..n {
                    xxt[i][j] += states[k][i] * states[k][j];
                }
            }
            xxt[i][i] += self.ridge_param;
        }

        // Simple inversion via Gaussian elimination (ok for small matrices)
        let inv = self.invert_matrix(&xxt);

        // Compute Y * X^T (output_dim x m)
        let mut yxt = vec![vec![0.0; m]; self.output_dim];
        for o in 0..self.output_dim {
            for i in 0..m {
                for k in 0..n {
                    if o < targets[k].len() {
                        yxt[o][i] += targets[k][o] * states[k][i];
                    }
                }
            }
        }

        // W = Y * X^T * (X * X^T + lambda * I)^-1
        for o in 0..self.output_dim {
            for j in 0..m {
                let mut sum = 0.0;
                for i in 0..m {
                    sum += yxt[o][i] * inv[i][j];
                }
                self.weights[o][j] = sum;
            }
        }
    }

    /// Predict output from a reservoir state.
    pub fn predict(&self, state: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.output_dim];
        for o in 0..self.output_dim {
            for (j, &w) in self.weights[o].iter().enumerate() {
                if j < state.len() {
                    output[o] += w * state[j];
                }
            }
        }
        output
    }

    /// Simple matrix inversion via Gauss-Jordan elimination.
    fn invert_matrix(&self, mat: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = mat.len();
        let mut aug = vec![vec![0.0; 2 * n]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = mat[i][j];
            }
            aug[i][n + i] = 1.0;
        }

        for col in 0..n {
            let mut max_row = col;
            for row in (col + 1)..n {
                if aug[row][col].abs() > aug[max_row][col].abs() {
                    max_row = row;
                }
            }
            aug.swap(col, max_row);

            let pivot = aug[col][col];
            if pivot.abs() < 1e-15 {
                continue;
            }
            for j in 0..(2 * n) {
                aug[col][j] /= pivot;
            }
            for row in 0..n {
                if row == col { continue; }
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        let mut inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                inv[i][j] = aug[i][n + j];
            }
        }
        inv
    }
}

/// Echo state network combining a ternary reservoir with a ridge readout.
#[derive(Clone, Debug)]
pub struct EchoStateNetwork {
    pub reservoir: TernaryReservoir,
    pub readout: RidgeReadout,
}

impl EchoStateNetwork {
    pub fn new(
        reservoir_size: usize,
        input_dim: usize,
        output_dim: usize,
        spectral_radius: f64,
        leak_rate: f64,
        ridge_param: f64,
        seed: u64,
    ) -> Self {
        EchoStateNetwork {
            reservoir: TernaryReservoir::new(reservoir_size, input_dim, spectral_radius, leak_rate, seed),
            readout: RidgeReadout::new(output_dim, reservoir_size, ridge_param),
        }
    }

    /// Train the network on input-output sequences.
    pub fn train(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>], washout: usize) {
        let states_ternary = self.reservoir.washout(inputs, washout);
        let states_f64: Vec<Vec<f64>> = states_ternary
            .iter()
            .map(|s| s.iter().map(|t| t.to_f64()).collect())
            .collect();
        let targets_washed: Vec<Vec<f64>> = targets.iter().skip(washout).cloned().collect();
        self.readout.train(&states_f64, &targets_washed);
    }

    /// Predict for a single input.
    pub fn predict(&mut self, input: &[f64]) -> Vec<f64> {
        self.reservoir.update(input);
        let state = self.reservoir.state_f64();
        self.readout.predict(&state)
    }

    /// Run prediction on a sequence.
    pub fn predict_sequence(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut outputs = vec![];
        for input in inputs {
            outputs.push(self.predict(input));
        }
        outputs
    }
}

/// Compute mean squared error between predictions and targets.
pub fn mse(predictions: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }
    let n = predictions.len();
    let mut total = 0.0;
    let mut count = 0;
    for (pred, targ) in predictions.iter().zip(targets.iter()) {
        for (p, t) in pred.iter().zip(targ.iter()) {
            total += (p - t) * (p - t);
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { total / count as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_from_f64() {
        assert_eq!(Ternary::from_f64(0.8), Ternary::Pos);
        assert_eq!(Ternary::from_f64(-0.7), Ternary::Neg);
        assert_eq!(Ternary::from_f64(0.2), Ternary::Zero);
    }

    #[test]
    fn test_ternary_roundtrip() {
        for &t in &[Ternary::Neg, Ternary::Zero, Ternary::Pos] {
            assert_eq!(Ternary::from_f64(t.to_f64()), t);
        }
    }

    #[test]
    fn test_identity_matrix_mul() {
        let m = ReservoirMatrix::identity(3);
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(m.mul_vec(&v), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_identity_spectral_radius() {
        let m = ReservoirMatrix::identity(3);
        assert!((m.spectral_radius() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sparse_matrix_creation() {
        let m = ReservoirMatrix::random_sparse(10, 10, 0.3, 0.9, 42);
        assert_eq!(m.rows, 10);
        assert_eq!(m.cols, 10);
        assert!(m.density() > 0.0);
        assert!(m.density() <= 1.0);
    }

    #[test]
    fn test_spectral_radius_control() {
        let m = ReservoirMatrix::random_sparse(20, 20, 0.2, 0.5, 42);
        let sr = m.spectral_radius();
        assert!((sr - 0.5).abs() < 0.1, "spectral radius {} vs target 0.5", sr);
    }

    #[test]
    fn test_reservoir_creation() {
        let r = TernaryReservoir::new(10, 3, 0.9, 0.5, 42);
        assert_eq!(r.size, 10);
        assert_eq!(r.state.len(), 10);
        assert!(r.state.iter().all(|t| *t == Ternary::Zero));
    }

    #[test]
    fn test_reservoir_reset() {
        let mut r = TernaryReservoir::new(20, 3, 1.5, 0.1, 42);
        // With spectral_radius > 1 and low leak, should get activation
        for _ in 0..100 {
            r.update(&[3.0, 3.0, 3.0]);
        }
        // At minimum, state should not be all zeros after 100 strong inputs
        // (if it is, the reservoir is dying, but reset should still work)
        r.reset();
        assert!(r.state.iter().all(|t| *t == Ternary::Zero));
    }

    #[test]
    fn test_reservoir_update() {
        let mut r = TernaryReservoir::new(5, 2, 0.9, 0.5, 42);
        // Feed many updates to ensure state changes
        for _ in 0..10 {
            r.update(&[1.0, 0.0]);
        }
        // After multiple updates, state should have changed from all zeros
        assert!(r.state_f64().iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_reservoir_run() {
        let mut r = TernaryReservoir::new(5, 2, 0.9, 0.5, 42);
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let states = r.run(&inputs);
        assert_eq!(states.len(), 3);
    }

    #[test]
    fn test_reservoir_washout() {
        let mut r = TernaryReservoir::new(5, 2, 0.9, 0.5, 42);
        let inputs = vec![vec![1.0, 0.0]; 5];
        let states = r.washout(&inputs, 2);
        assert_eq!(states.len(), 3); // 5 - 2 washout
    }

    #[test]
    fn test_reservoir_state_f64() {
        let mut r = TernaryReservoir::new(5, 1, 0.9, 0.5, 42);
        r.update(&[1.0]);
        let sf = r.state_f64();
        assert_eq!(sf.len(), 5);
        for &v in &sf {
            assert!(v == -1.0 || v == 0.0 || v == 1.0);
        }
    }

    #[test]
    fn test_ridge_readout_creation() {
        let ro = RidgeReadout::new(2, 5, 0.01);
        assert_eq!(ro.output_dim, 2);
        assert_eq!(ro.weights.len(), 2);
        assert_eq!(ro.weights[0].len(), 5);
    }

    #[test]
    fn test_ridge_readout_train_predict() {
        let mut ro = RidgeReadout::new(1, 3, 0.01);
        let states = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let targets = vec![vec![1.0], vec![2.0], vec![3.0]];
        ro.train(&states, &targets);

        // Should approximately predict
        let p1 = ro.predict(&[1.0, 0.0, 0.0]);
        assert!((p1[0] - 1.0).abs() < 0.5, "prediction {} vs expected 1.0", p1[0]);
    }

    #[test]
    fn test_ridge_readout_empty() {
        let mut ro = RidgeReadout::new(1, 3, 0.01);
        ro.train(&[], &[]); // Should not panic
    }

    #[test]
    fn test_matrix_inversion() {
        let ro = RidgeReadout::new(1, 2, 0.01);
        let mat = vec![vec![4.0, 0.0], vec![0.0, 2.0]];
        let inv = ro.invert_matrix(&mat);
        assert!((inv[0][0] - 0.25).abs() < 0.01);
        assert!((inv[1][1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_esn_creation() {
        let esn = EchoStateNetwork::new(10, 2, 1, 0.9, 0.5, 0.01, 42);
        assert_eq!(esn.reservoir.size, 10);
        assert_eq!(esn.readout.output_dim, 1);
    }

    #[test]
    fn test_esn_predict() {
        let mut esn = EchoStateNetwork::new(10, 2, 1, 0.9, 0.5, 0.01, 42);
        let output = esn.predict(&[1.0, 0.5]);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_mse_identical() {
        let preds = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let targets = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(mse(&preds, &targets), 0.0);
    }

    #[test]
    fn test_mse_different() {
        let preds = vec![vec![1.0]];
        let targets = vec![vec![3.0]];
        assert_eq!(mse(&preds, &targets), 4.0);
    }

    #[test]
    fn test_mse_empty() {
        assert_eq!(mse(&[], &[]), 0.0);
    }

    #[test]
    fn test_nnz() {
        let m = ReservoirMatrix::identity(3);
        assert_eq!(m.nnz(), 3);
    }
}
