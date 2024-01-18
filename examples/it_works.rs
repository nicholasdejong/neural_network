use nalgebra::DVector;
use neural_network::{batch, functions::Activation, neural_network::NeuralNetwork};

fn main() {
    let mut nn = NeuralNetwork::new(&[2, 3, 1]);
    // nn.set_activations(vec![Activation::ReLU, Activation::SoftPlus]);
    // dbg!(&nn.weights);
    // dbg!(&nn.biases);

    let input = DVector::from_vec(vec![-1.0, 0.2]);
    // let output = nn.forward(input.clone());
    let layers = nn.forward_layers(input);

    let data = batch![
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0])
    ];

    nn.backward(&data);

    // println!("out: {output:?}");
    dbg!(layers);
}
