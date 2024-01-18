use na::DVector;

pub trait CostFunction {
    /// Used to determine how accurate the Neural Network is with respect to the training data.
    fn forward(&self, observed: DVector<f32>, training: DVector<f32>) -> DVector<f32>;
    /// The derived `forward` implentation used during backpropagation.
    fn backward(&self, observed: DVector<f32>, training: DVector<f32>) -> DVector<f32>;
}

pub enum Cost {
    SquaredResiduals,
    Custom(Box<dyn CostFunction>),
}

impl Cost {
    pub fn forward(&self, observed: DVector<f32>, training: DVector<f32>) -> DVector<f32> {
        match self {
            Self::SquaredResiduals => SquaredResiduals.forward(observed, training),
            Self::Custom(f) => f.forward(observed, training),
        }
    }

    pub fn backward(&self, observed: DVector<f32>, training: DVector<f32>) -> DVector<f32> {
        match self {
            Self::SquaredResiduals => SquaredResiduals.backward(observed, training),
            Self::Custom(f) => f.backward(observed, training),
        }
    }
}

pub struct SquaredResiduals;

impl CostFunction for SquaredResiduals {
    fn forward(&self, observed: DVector<f32>, training: DVector<f32>) -> DVector<f32> {
        (&observed - &training) * (&observed - &training)
    }

    fn backward(&self, observed: DVector<f32>, training: DVector<f32>) -> DVector<f32> {
        2.0 * (&observed - &training)
    }
}

pub trait ActivationFunction {
    // Used to 'activate' neurons during forward propagation.
    fn forward(&self, l: DVector<f32>) -> DVector<f32>;
    // The derived `forward` implementation used during backpropagation.
    fn backward(&self, l: DVector<f32>) -> DVector<f32>;
}

pub enum Activation {
    /// Does not activate the layer.
    Inactive,
    /// Uses a [Rectified](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) Linear Unit to activate the layer.
    ReLU,
    SoftPlus,
    Custom(Box<dyn ActivationFunction>),
}

impl Activation {
    pub fn forward(&self, l: DVector<f32>) -> DVector<f32> {
        match &self {
            Activation::Inactive => l,
            Activation::ReLU => ReLU.forward(l),
            Activation::SoftPlus => SoftPlus.forward(l),
            Activation::Custom(f) => f.forward(l),
        }
    }

    pub fn backward(&self, l: DVector<f32>) -> DVector<f32> {
        match &self {
            Activation::Inactive => DVector::from_vec(l.iter().map(|_| 1.0).collect::<Vec<f32>>()),
            Activation::ReLU => ReLU.backward(l),
            Activation::SoftPlus => SoftPlus.backward(l),
            Activation::Custom(f) => f.backward(l),
        }
    }
}

/// A [Rectified](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) Linear Unit implementation.
pub struct ReLU;

impl ActivationFunction for ReLU {
    fn forward(&self, l: DVector<f32>) -> DVector<f32> {
        DVector::from_vec(l.iter().map(|v| v.max(0.0)).collect::<Vec<f32>>())
    }

    fn backward(&self, l: DVector<f32>) -> DVector<f32> {
        DVector::from_vec(
            l.iter()
                .map(|v| 0.5 + 0.5 * v.signum())
                .collect::<Vec<f32>>(),
        )
    }
}

pub struct SoftPlus;

impl ActivationFunction for SoftPlus {
    fn forward(&self, l: DVector<f32>) -> DVector<f32> {
        DVector::from_vec(l.iter().map(|v| (1.0 + v.exp()).ln()).collect::<Vec<f32>>())
    }

    fn backward(&self, l: DVector<f32>) -> DVector<f32> {
        DVector::from_vec(
            l.iter()
                .map(|v| 1.0 / (1.0 + (-v).exp()))
                .collect::<Vec<f32>>(),
        )
    }
}
