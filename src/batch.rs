pub struct Sample(pub Vec<f32>, pub Vec<f32>);

impl Sample {
    pub fn new(input: Vec<f32>, output: Vec<f32>) -> Self {
        Self(input, output)
    }
}

/// Batches store data that is used to train the neural network.
pub struct Batch {
    pub samples: Vec<Sample>,
}

impl Batch {
    pub fn new(samples: Vec<Sample>) -> Self {
        Self { samples }
    }
}

#[macro_export]
macro_rules! batch {
    ($(($in: expr, $out: expr)),*) => {{
        use crate::batch::{Sample, Batch};
        let mut samples = vec![];
        $(
            samples.push(Sample::new($in.to_vec(), $out.to_vec()));
        )*
        let batch = Batch::new(samples);
        batch
    }};
}
