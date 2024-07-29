use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use ndarray::AssignElem;

pub fn stride_from_shape(shape: &[usize], device: &Device) -> Result<Tensor> {
    let mut stride_values = vec![1];
    for &x in shape[1..].iter().rev() {
        let last = *stride_values.last().unwrap();
        stride_values.push(last * x as u32);
    }
    stride_values.reverse();
    // Assuming the library requires specifying the data type explicitly when creating a tensor
    let stride_tensor = Tensor::from_vec(stride_values.clone(), &[stride_values.len()], device)?
        .to_dtype(DType::I64)?;
    Ok(stride_tensor)
}

pub fn scatter_add_nd(input: &mut Tensor, indices: &Tensor, values: &Tensor) -> Result<()> {
    let d = indices.dims()[indices.dims().len() - 1];
    let c = input.dims()[input.dims().len() - 1];
    let size = &input.shape().dims()[..(input.shape().dims().len() - 1)];

    // let size = &input.shape().dims()[..input.shape().dims().len() - 1];
    let stride = stride_from_shape(size, &input.device())?;

    assert_eq!(size.len(), d);

    let num_elements: usize = input.elem_count() / c;
    let flat_input = input.reshape(vec![num_elements, c])?;
    let flatten_indices = indices.mul(&stride)?.sum(D::Minus1)?;

    // First, ensure flat_indices is prepared correctly:
    let expanded_indices = flatten_indices.unsqueeze(1)?.repeat(&[1, c]).unwrap();

    // Now, perform the scatter_add operation:
    let updated_tensor = flat_input.scatter_add(&expanded_indices, values, 0)?;

    // vn = vn.scatter_add(&i0.unsqueeze(1)?.repeat(&[1, 3])?, &face_normals, 0)?;

    *input = updated_tensor.reshape(size)?;
    Ok(())
}

fn scatter_add_nd_with_count(
    input: &Tensor,
    count: &Tensor,
    indices: &Tensor,
    values: &Tensor,
    weights: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let d = indices.dims()[indices.dims().len() - 1];
    let c = input.dims()[input.dims().len() - 1];
    let size = &input.shape().dims()[..input.shape().dims().len() - 1];
    let stride = stride_from_shape(size, &input.device())?;

    let num_elements: usize = input.elem_count() / c;

    assert_eq!(size.len(), d);

    let weights = match weights {
        Some(w) => w.clone(),
        None => Tensor::ones_like(&values.i((.., 0))?)?,
    };

    let flat_indices = indices.mul(&stride)?.sum(D::Minus1)?;
    let repeated_indices = flat_indices.unsqueeze(1)?.repeat(vec![1, c])?;

    let flat_input = input.reshape(vec![num_elements, c])?;
    let input_updated = flat_input.scatter_add(&repeated_indices, values, 0)?;
    let input_reshaped =
        input_updated.reshape(size.iter().chain(&[c]).cloned().collect::<Vec<_>>())?;

    let count_flat = count.reshape(vec![num_elements, 1])?;
    let count_updated = count_flat.scatter_add(&flat_indices.unsqueeze(1)?, &weights, 0)?;
    let count_reshaped =
        count_updated.reshape(size.iter().chain(&[1]).cloned().collect::<Vec<_>>())?;

    Ok((input_reshaped, count_reshaped))
}

fn mipmap_linear_grid_put_2d(
    h: usize,
    w: usize,
    coords: &Tensor,
    values: &Tensor,
    min_resolution: usize,
    return_count: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    let c = values.dims()[values.dims().len() - 1];
    // let c = values.shape()[-1];
    let mut result = Tensor::zeros(&[h, w, c], values.dtype(), values.device())?;
    let mut count = Tensor::zeros(&[h, w, 1], values.dtype(), values.device())?;

    let mut cur_h = h;
    let mut cur_w = w;

    while std::cmp::min(cur_h, cur_w) > min_resolution {
        let mask = count.squeeze(D::Minus1)?.eq(0 as i64);

        match mask {
            Ok(mask) => {
                let (cur_result, cur_count) =
                    linear_grid_put_2d(cur_h, cur_w, coords, values, true)?;

                // TODO: Interpolation and updating result and count tensors

                // let interpolated_result = cur_result.interpolate(&[h, w], "bilinear", false)?;
                // let interpolated_count = cur_count.interpolate(&[h, w], "bilinear", false)?;

                cur_h /= 2;
                cur_w /= 2;
            }
            Err(_) => break,
        }
    }

    if return_count {
        Ok((result, Some(count)))
    } else {
        apply_masked_division(&mut result, &count, c, D::Minus1)?;
        Ok((result, None))
    }
}

fn linear_grid_put_2d(
    h: usize,
    w: usize,
    coords: &Tensor,
    values: &Tensor,
    return_count: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    let c = values.dims()[values.dims().len() - 1];
    // let c = values.shape()[-1];

    // Create a scaling tensor for the coordinates transformation
    let scale_tensor = Tensor::new(vec![h as f32 - 1.0, w as f32 - 1.0], &coords.device())?;

    let indices = (((coords * 0.5)? + 0.5) * scale_tensor)?
        .round()?
        .to_dtype(DType::I64)?;

    let indices_00 = indices.floor()?.to_dtype(DType::I64)?;

    indices_00.i((.., 0))?.clamp(0 as i64, h as i64 - 2)?;
    indices_00.i((.., 1))?.clamp(0 as i64, w as i64 - 2)?;

    let indices_01 = (&indices_00 + Tensor::new(vec![0 as i64, 1 as i64], indices.device()))?;

    let indices_10 = (&indices_00 + Tensor::new(vec![1 as i64, 0 as i64], indices.device()))?;

    let indices_11 = (&indices_00 + Tensor::new(vec![1 as i64, 1 as i64], indices.device()))?;

    let h_tensor = (&indices.i((.., 0))? - indices_00.i((.., 0))?.to_dtype(DType::F32))?;
    let w_tensor = (&indices.i((.., 1))? - indices_00.i((.., 1))?.to_dtype(DType::F32))?;
    let w_00 = ((1.0 - &h_tensor)? * (1.0 - &w_tensor)?)?;
    let w_01 = ((1.0 - &h_tensor) * &w_tensor)?;
    let w_10 = (&h_tensor * (1.0 - &w_tensor))?;
    let w_11 = (&h_tensor * &w_tensor)?;

    let mut result = Tensor::zeros((h as usize, w as usize, c), values.dtype(), values.device())?;
    let mut count = Tensor::zeros((h as usize, w as usize, 1), values.dtype(), values.device())?;

    scatter_add_nd_with_count(
        &mut result,
        &mut count,
        &indices_00,
        &(values * w_00.unsqueeze(D::Minus1))?,
        Some(&(w_00.unsqueeze(D::Minus1))?),
    )?;

    scatter_add_nd_with_count(
        &mut result,
        &mut count,
        &indices_01,
        &(values * w_01.unsqueeze(D::Minus1))?,
        Some(&(w_01.unsqueeze(D::Minus1))?),
    )?;

    scatter_add_nd_with_count(
        &mut result,
        &mut count,
        &indices_10,
        &(values * w_10.unsqueeze(D::Minus1))?,
        Some(&(w_10.unsqueeze(D::Minus1))?),
    )?;

    scatter_add_nd_with_count(
        &mut result,
        &mut count,
        &indices_11,
        &(values * w_11.unsqueeze(D::Minus1))?,
        Some(&(w_11.unsqueeze(D::Minus1))?),
    )?;

    if return_count {
        Ok((result, Some(count)))
    } else {
        apply_masked_division(&mut result, &count, c, D::Minus1)?;

        Ok((result, None))
    }
}

fn apply_masked_division(result: &mut Tensor, count: &Tensor, c: usize, dim: D) -> Result<()> {
    let mask = count.squeeze(dim)?.gt(0 as i64)?;
    let repeated_count = count.i(&mask)?.repeat(&[1, c])?;
    let masked_result = result.i(&mask)?;

    let updated_result = masked_result.div(&repeated_count)?;

    // result.i(&mask)?.gather(&updated_result, D::Minus1)?;
    result.i(&mask)?.assign_elem(updated_result);

    Ok(())
}
