use candle_core::{DType, Device, IndexOp, Result, Tensor};
use ndarray::{s, Array2, AssignElem};
use ndarray_linalg::Inverse;
use std::f32::consts::PI;

use crate::array_utils::ndarray_to_tensor;

// use crate::sh_utils::*;
// use crate::mesh_utils::*;

pub fn inverse_sigmoid(x: &Tensor) -> Result<Tensor> {
    let log_base = (x / (1.0 - x))?;
    Ok(log_base.log()?)
}

pub fn get_expon_lr_func(
    lr_init: f32,
    lr_final: f32,
    lr_delay_steps: Option<usize>,
    lr_delay_mult: Option<f32>,
    max_steps: Option<usize>,
) -> Box<dyn Fn(usize) -> f32> {
    let lr_delay_steps = lr_delay_steps.unwrap_or(0);
    let lr_delay_mult = lr_delay_mult.unwrap_or(1.0);
    let max_steps = max_steps.unwrap_or(1000000);

    Box::new(move |step: usize| -> f32 {
        if lr_init == lr_final {
            return lr_init;
        }
        if step == 0 || (lr_init == 0.0 && lr_final == 0.0) {
            return 0.0;
        }
        let delay_rate = if lr_delay_steps > 0 {
            lr_delay_mult
                + (1.0 - lr_delay_mult)
                    * ((0.5 * PI * (step as f32 / lr_delay_steps as f32).min(1.0)).sin())
        } else {
            1.0
        };
        let t = (step as f32 / max_steps as f32).min(1.0);
        let log_lerp = ((lr_init.ln() * (1.0 - t)) + (lr_final.ln() * t)).exp();
        delay_rate * log_lerp
    })
}

// Define the strip_lowerdiag function
pub fn strip_lowerdiag(l: &Tensor) -> Result<Tensor> {
    // let device = l.device();
    let device = &Device::new_cuda(0).unwrap();

    let uncertainty = Tensor::zeros((l.dims()[0], 6), DType::F32, device)?;

    uncertainty.i((.., 0))?.assign_elem(l.i((.., 0, 0))?);
    uncertainty.i((.., 1))?.assign_elem(l.i((.., 0, 1))?);
    uncertainty.i((.., 2))?.assign_elem(l.i((.., 0, 2))?);
    uncertainty.i((.., 3))?.assign_elem(l.i((.., 1, 1))?);
    uncertainty.i((.., 4))?.assign_elem(l.i((.., 1, 2))?);
    uncertainty.i((.., 5))?.assign_elem(l.i((.., 2, 2))?);

    Ok(uncertainty)
}

pub fn strip_symmetric(sym: &Tensor) -> Result<Tensor> {
    strip_lowerdiag(sym)
}

pub fn gaussian_3d_coeff(xyzs: &Tensor, covs: &Tensor) -> Result<Tensor> {
    let x = xyzs.i((.., 0))?;
    let y = xyzs.i((.., 1))?;
    let z = xyzs.i((.., 2))?;

    let a = covs.i((.., 0))?;
    let b = covs.i((.., 1))?;
    let c = covs.i((.., 2))?;
    let d = covs.i((.., 3))?;
    let e = covs.i((.., 4))?;
    let f = covs.i((.., 5))?;

    let e_square = (&e * &e)?;
    let b_square = (&b * &b)?;
    let c_square = (&c * &c)?;

    let a_d = (&a * &d)?;
    let e_c = (&e * &c)?;

    let inv_det = (1.0
        / ((((&a_d * &f)? + 2.0 * &e_c * &b)?
            - (&e_square * &a)?
            - (&c_square * &d)?
            - (&b_square * &f)?)?
            + 1e-24)?)?;
    let inv_a = ((&d * &f - &e_square) * &inv_det)?;
    let inv_b = ((&e_c - &b * &f) * &inv_det)?;
    let inv_c = (((&e * &b)? - (&c * &d)?) * &inv_det)?;
    let inv_d = ((&a * &f - &c_square) * &inv_det)?;
    let inv_e = (((&b * &c)? - (&e * &a)?) * &inv_det)?;
    let inv_f = ((&a * &d - &b_square) * &inv_det)?;

    let power = (-0.5 * (((&x * &x) * &inv_a)? + ((&y * &y) * &inv_d)? + ((&z * &z) * &inv_f)?)?
        - (&x * &y * &inv_b)?
        - (&x * &z * &inv_c)?
        - (&y * &z * &inv_e)?)?;

    // TODO - translate from pytorch
    // power[power > 0] = -1e10 # abnormal values... make weights 0
    // power.apply(|x| if x > 0.0 { -1e10 } else { x });
    Ok(power.exp()?)
}

// Function to build rotation matrix from quaternion
pub fn build_rotation(r: &Tensor) -> Result<Tensor> {
    let r_0_square = (r.i((.., 0))? * r.i((.., 0))?)?;
    let r_1_square = (r.i((.., 1))? * r.i((.., 1))?)?;
    let r_2_square = (r.i((.., 2))? * r.i((.., 2))?)?;
    let r_3_square = (r.i((.., 3))? * r.i((.., 3))?)?;

    let norm = (r_0_square + r_1_square + r_2_square + r_3_square)?.sqrt()?;
    let q = (r / norm)?;

    let r = q.i((.., 0))?;
    let x = q.i((.., 1))?;
    let y = q.i((.., 2))?;
    let z = q.i((.., 3))?;

    let result = Tensor::zeros((q.dims()[0], 3, 3), DType::F32, r.device())?;

    result
        .i((.., 0, 0))?
        .assign_elem((1.0 - (2.0 * ((&y * &y)? + (&z * &z)?)?)?)?);
    result
        .i((.., 0, 1))?
        .assign_elem((2.0 * ((&x * &y)? - (&r * &z)?)?)?);
    result
        .i((.., 0, 2))?
        .assign_elem((2.0 * ((&x * &z)? + (&r * &y)?)?)?);
    result
        .i((.., 1, 0))?
        .assign_elem((2.0 * ((&x * &y)? + (&r * &z)?)?)?);
    result
        .i((.., 1, 1))?
        .assign_elem((1.0 - (2.0 * ((&x * &x)? + (&z * &z)?)?)?)?);
    result
        .i((.., 1, 2))?
        .assign_elem((2.0 * ((&y * &z)? - (&r * &x)?)?)?);
    result
        .i((.., 2, 0))?
        .assign_elem((2.0 * ((&x * &z)? - (&r * &y)?)?)?);
    result
        .i((.., 2, 1))?
        .assign_elem((2.0 * ((&y * &z)? + (&r * &x)?)?)?);
    result
        .i((.., 2, 2))?
        .assign_elem((1.0 - (2.0 * ((&x * &x)? + (&y * &y)?)?)?)?);

    Ok(result)
}

// Function to build scaling rotation matrix
pub fn build_scaling_rotation(s: &Tensor, r: &Tensor) -> Result<Tensor> {
    let l = Tensor::zeros((s.dims()[0], 3, 3), DType::F32, s.device())?;
    let result = build_rotation(r)?;

    l.i((.., 0, 0))?.assign_elem(s.i((.., 0))?);
    l.i((.., 1, 1))?.assign_elem(s.i((.., 1))?);
    l.i((.., 2, 2))?.assign_elem(s.i((.., 2))?);

    Ok(result.matmul(&l)?)
}

// Function to calculate the projection matrix for a camera
pub fn get_projection_matrix(znear: f32, zfar: f32, fovx: f32, fovy: f32) -> Result<Tensor> {
    let tan_half_fovy = (fovy / 2.0).tan();
    let tan_half_fovx = (fovx / 2.0).tan();

    let mut p = Tensor::zeros((4, 4), DType::F32, &Device::Cpu)?;

    let z_sign = 1.0;

    // TODO - assign the float values to tensor
    // p.i((.., 0, 0))?.assign_elem(1.0 / tan_half_fovx);
    // p.i((.., 1, 1))?.assign_elem(1.0 / tan_half_fovy);
    // p.i((.., 3, 2))?.assign_elem(z_sign);
    // p.i((.., 2, 2))?.assign_elem(z_sign * zfar / (zfar - znear));
    // p.i((.., 2, 3))?.assign_elem(-(zfar * znear) / (zfar - znear));

    Ok(p)
}

pub struct MiniCam {
    image_width: usize,
    image_height: usize,
    fovy: f32,
    fovx: f32,
    znear: f32,
    zfar: f32,
    world_view_transform: Tensor,
    projection_matrix: Tensor,
    full_proj_transform: Tensor,
    camera_center: Tensor,
}

impl MiniCam {
    pub fn new(
        c2w: Array2<f32>,
        width: usize,
        height: usize,
        fovy: f32,
        fovx: f32,
        znear: f32,
        zfar: f32,
    ) -> Self {
        let mut w2c = c2w.inv().expect("Matrix inversion failed");

        // Rectify the matrix
        for i in 1..3 {
            for j in 0..3 {
                w2c[[i, j]] *= -1.0;
            }
        }
        for j in 0..3 {
            w2c[[j, 3]] *= -1.0;
        }

        let device = &Device::new_cuda(0).unwrap();
        // Convert ndarray to Tensor manually
        let world_view_transform = ndarray_to_tensor(&w2c, &device).unwrap();
        let projection_matrix = get_projection_matrix(znear, zfar, fovx, fovy)
            .unwrap()
            .transpose(0, 1)
            .unwrap()
            .to_device(device)
            .unwrap();
        let full_proj_transform = world_view_transform.matmul(&projection_matrix).unwrap();
        let camera_center = -1.0
            * ndarray_to_tensor(
                &c2w.slice(s![..3, 3]).to_owned().into_shape((3, 1)).unwrap(),
                &device,
            )
            .unwrap();

        MiniCam {
            image_width: width,
            image_height: height,
            fovy,
            fovx,
            znear,
            zfar,
            world_view_transform,
            projection_matrix,
            full_proj_transform,
            camera_center: camera_center.unwrap(),
        }
    }
}

struct BasicPointCloud {
    points: Array2<f32>,
    colors: Array2<f32>,
    normals: Array2<f32>,
}

struct GaussianModel {
    active_sh_degree: i32,
    max_sh_degree: i32,
    xyz: Tensor,
    features_dc: Tensor,
    features_rest: Tensor,
    scaling: Tensor,
    rotation: Tensor,
    opacity: Tensor,
    max_radii2d: Tensor,
    xyz_gradient_accum: Tensor,
    denom: Tensor,
    percent_dense: f32,
    spatial_lr_scale: f32,
}

impl GaussianModel {
    pub fn new(sh_degree: i32) -> Self {
        GaussianModel {
            active_sh_degree: 0,
            max_sh_degree: sh_degree,
            xyz: Tensor::zeros(0, DType::F32, &Device::Cpu).unwrap(),
            features_dc: Tensor::zeros(0, DType::F32, &Device::Cpu).unwrap(),
            features_rest: Tensor::zeros(0, DType::F32, &Device::Cpu).unwrap(),
            scaling: Tensor::zeros(0, DType::F32, &Device::Cpu).unwrap(),
            rotation: Tensor::zeros(0, DType::F32, &Device::Cpu).unwrap(),
            opacity: Tensor::zeros(0, DType::F32, &Device::Cpu).unwrap(),
            max_radii2d: Tensor::zeros(0, DType::F32, &Device::Cpu).unwrap(),
            xyz_gradient_accum: Tensor::zeros(0, DType::F32, &Device::Cpu).unwrap(),
            denom: Tensor::zeros(0, DType::F32, &Device::Cpu).unwrap(),
            percent_dense: 0.0,
            spatial_lr_scale: 0.0,
        }
    }

    pub fn setup_functions(&mut self) {
        !unimplemented!()
    }

    pub fn capture(
        &self,
    ) -> (
        i32,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        f32,
    ) {
        (
            self.active_sh_degree,
            self.xyz.clone(),
            self.features_dc.clone(),
            self.features_rest.clone(),
            self.scaling.clone(),
            self.rotation.clone(),
            self.opacity.clone(),
            self.max_radii2d.clone(),
            self.xyz_gradient_accum.clone(),
            self.denom.clone(),
            // self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    }
}
