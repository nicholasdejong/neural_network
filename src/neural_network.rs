use na::{self, DMatrix, DVector, Vector1};

use crate::{
    batch::{Batch, Sample},
    functions::{Activation, ActivationFunction, Cost, CostFunction, ReLU, SoftPlus},
    helpers::multiply,
};

pub struct NeuralNetwork {
    pub weights: Vec<DMatrix<f32>>,
    pub biases: Vec<DVector<f32>>,
    input_size: usize,
    output_size: usize,
    activations: Vec<Activation>,
    cost: Cost,
}

impl NeuralNetwork {
    /// Defines the size of each layer of the new Neural Network.
    /// The first and last values of `layers` represent the input and output layer sizes respectively.
    pub fn new(layers: &[usize]) -> Self {
        assert!(!layers.contains(&0), "Layers must have non-zero size.");
        assert!(
            layers.len() >= 2,
            "Neural Network must have at least 2 layers."
        );
        let mut weights: Vec<DMatrix<f32>> = vec![];
        for (i, layer) in layers.iter().enumerate() {
            if i == 0 {
                continue;
            }
            weights.push(DMatrix::new_random(*layer, layers[i - 1]));
        }
        let biases: Vec<DVector<f32>> = layers[1..]
            .iter()
            .map(|l| DVector::new_random(*l))
            .collect();
        let mut activations = vec![];
        activations.resize_with(layers.len(), || Activation::Inactive);
        Self {
            weights,
            biases,
            input_size: *layers.first().unwrap(),
            output_size: *layers.last().unwrap(),
            activations,
            cost: Cost::SquaredResiduals,
        }
    }

    pub fn set_activations(&mut self, activations: Vec<Activation>) {
        assert_eq!(
            activations.len(),
            self.weights.len(),
            "Expected {} activations, found {}.",
            self.weights.len(),
            activations.len()
        );
        self.activations = activations;
    }

    /// Feeds forward `n` layers after the input layer. `n` <= no. of hidden layers + 1
    fn forward_n(&self, input: DVector<f32>, n: usize) -> DVector<f32> {
        let mut current = input.clone();
        for i in 0..n {
            current = self.weights[i].clone() * current;
            current += self.biases[i].clone();
            current = self.activations[i].forward(current);
        }
        current
    }

    pub fn forward(&self, input: DVector<f32>) -> DVector<f32> {
        self.forward_n(input, self.weights.len())
    }

    /// Returns all (unactivated) layers instead of just the output layer
    pub fn forward_layers(&self, input: DVector<f32>) -> Vec<DVector<f32>> {
        let mut current = input.clone();
        let mut layers = vec![input];
        for i in 0..self.weights.len() {
            current = self.weights[i].clone() * current;
            current += self.biases[i].clone();
            layers.push(current.clone());
            current = self.activations[i].forward(current);
        }
        layers
    }

    // fn cost(&self, data: &[(&[f32], &[f32])]) -> DVector<f32> {
    //     let mut cumul_cost = DVector::<f32>::zeros(self.output_size);
    //     for (input, output) in data {
    //         let observed = self.forward(DVector::from_row_slice(*input));
    //         let cost = self
    //             .cost
    //             .forward(observed, DVector::from_row_slice(*output));
    //         cumul_cost += cost;
    //     }
    //     cumul_cost / data.len() as f32
    // }

    pub fn backward(&mut self, data: &Batch) -> DVector<f32> {
        let mut weights_changes = vec![];
        for weight in &self.weights {
            weights_changes.push(DMatrix::<f32>::zeros(
                weight.column(0).len(),
                weight.row(0).len(),
            ));
        }
        let mut bias_changes = vec![];
        for bias in &self.biases {
            bias_changes.push(DVector::<f32>::zeros(bias.column(0).len()))
        }
        // let _ = &self.weights[0] + &weights_changes[0];
        // let _ = &self.biases[0] + &bias_changes[0];
        for Sample(input, output) in &data.samples {
            let (input, output) = (
                DVector::from_row_slice(&input),
                DVector::from_row_slice(&output),
            );
            let layers = self.forward_layers(input);
            // wrt a
            let mut cost = self.cost.backward(
                self.activations
                    .last()
                    .unwrap()
                    .forward(layers.last().unwrap().clone()),
                output,
            );
            for i in (0..self.weights.len()).rev() {
                let wrt_z = self.activations[i].backward(layers[i].clone());
                cost.component_mul_assign(&wrt_z);
            }
        }
        // let output_wrt_z =
        // cost wrt output
        // each loop {
        // output wrt z -> b
        // * prev node output
        //
        //}

        todo!()
    }
}
