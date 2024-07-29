use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use ndarray::{Array, Array2, ArrayD, AssignElem, Axis, IxDyn};

fn dot(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    x.mul(y)?.sum(D::Minus1)
}

fn length(x: &Tensor, eps: f32) -> Result<Tensor> {
    dot(x, x)?.clamp(eps, f32::INFINITY)?.sqrt()
}

pub fn safe_normalize(x: &Tensor, eps: f32) -> Result<Tensor> {
    x.div(&length(x, eps)?)
}

fn nd_dot(x: &ArrayD<f32>, y: &ArrayD<f32>) -> Result<ArrayD<f32>> {
    let multiplied_array = x * y;
    let result = multiplied_array.sum_axis(Axis(multiplied_array.len() - 1));
    Ok(result)
}

fn nd_length(x: &ArrayD<f32>, eps: Option<f32>) -> Result<ArrayD<f32>> {
    let eps = eps.unwrap_or(f32::EPSILON);
    let array_square = x * x;
    let result = array_square
        .sum_axis(Axis(array_square.len() - 1))
        .map(|x| f32::max(*x, eps))
        .mapv(f32::sqrt);
    Ok(result)
}

pub fn nd_safe_normalize(x: &ArrayD<f32>, eps: Option<f32>) -> Result<ArrayD<f32>> {
    let eps = eps.unwrap_or(f32::EPSILON);
    let result = x / nd_length(x, Some(eps))?;
    Ok(result)
}

/// Computes the cross product of two 3D vectors.
pub fn cross_product(a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {
    assert_eq!(a.len(), 3, "First vector must be 3D");
    assert_eq!(b.len(), 3, "Second vector must be 3D");

    let cross_prod = Array::from_shape_vec(
        IxDyn(&[3]),
        vec![
            a[1] * b[2] - a[2] * b[1], // i
            a[2] * b[0] - a[0] * b[2], // j
            a[0] * b[1] - a[1] * b[0], // k
        ],
    )
    .expect("Failed to create ArrayD");

    cross_prod
}

pub fn ndarray_to_tensor(array: &Array2<f32>, device: &Device) -> Result<Tensor> {
    let dims = array.dim();

    let mut tensor = Tensor::zeros((dims.0 as usize, dims.1 as usize), DType::F32, device)?;

    for ((i, j), &value) in array.indexed_iter() {
        let value_vec = vec![value];

        let scalar_tensor = Tensor::from_vec(value_vec, (1,), device)?; // Create a single-element tensor

        // Assign the scalar tensor to the appropriate position in the main tensor
        tensor.i((i, j))?.assign_elem(scalar_tensor);
    }

    Ok(tensor)
}
