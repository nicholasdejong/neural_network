use neural_network::functions::Activation;
use neural_network::neural_network::NeuralNetwork;

// A Neural Network to solve an XOR gate.

fn main() {
    // // Our NN
    // let mut nn = NeuralNetwork::new(&[2, 3, 1])
    //     .with_activations(vec![Some(Box::new(ReLU)), Some(Box::new(ReLU))]);

    // // Our data to train the NN with.
    let data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    // let result = nn.train(data).learning_rate(0.03).begin();

    // result.cost();
}
