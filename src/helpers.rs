use nalgebra::DVector;

pub fn multiply(a: DVector<f32>, b: DVector<f32>) -> DVector<f32> {
    assert_eq!(a.len(), b.len());
    dbg!(a[0]);
    todo!()
}
