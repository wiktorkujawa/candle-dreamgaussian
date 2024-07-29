use candle_core::Result;
use nalgebra::{DMatrix, Matrix3, Rotation3, Unit, Vector3};
use ndarray::{
    arr2, array, s, stack, Array, Array1, Array2, ArrayBase, ArrayD, Axis, Dim, IxDyn,
    OwnedRepr,
};
use ndarray_linalg::Inverse;

use crate::array_utils::{cross_product, nd_safe_normalize};

fn nalgebra_to_ndarray(matrix: &DMatrix<f32>) -> Array2<f32> {
    Array2::from_shape_vec(
        (matrix.nrows(), matrix.ncols()),
        matrix.iter().cloned().collect(),
    )
    .unwrap()
}

fn ndarray_to_nalgebra(array: &Array2<f32>) -> DMatrix<f32> {
    DMatrix::from_iterator(array.nrows(), array.ncols(), array.iter().cloned())
}

fn look_at(campos: ArrayD<f32>, target: ArrayD<f32>, opengl: Option<bool>) -> ArrayD<f32> {
    // campos: [N, 3], camera/eye position
    // target: [N, 3], object to look at
    // # return: [N, 3, 3], rotation matrix

    let mut forward_vector: ArrayD<f32> =
        Array::from_shape_vec(IxDyn(&[3]), vec![0.0, 0.0, 0.0]).unwrap();
    let mut up_vector: ArrayD<f32> =
        Array::from_shape_vec(IxDyn(&[3]), vec![0.0, 1.0, 0.0]).unwrap();
    let mut right_vector: ArrayD<f32> =
        Array::from_shape_vec(IxDyn(&[3]), vec![0.0, 0.0, 0.0]).unwrap();

    let opengl = opengl.unwrap_or(true);
    if !opengl {
        // # camera forward aligns with -z
        forward_vector = nd_safe_normalize(&(target - campos), None).unwrap();
        right_vector =
            nd_safe_normalize(&(cross_product(&forward_vector, &up_vector)), None).unwrap();
        up_vector =
            nd_safe_normalize(&(cross_product(&right_vector, &forward_vector)), None).unwrap();
    } else {
        // # camera forward aligns with +z
        forward_vector = nd_safe_normalize(&(campos - target), None).unwrap();
        right_vector =
            nd_safe_normalize(&(cross_product(&up_vector, &forward_vector)), None).unwrap();
        up_vector =
            nd_safe_normalize(&(cross_product(&forward_vector, &right_vector)), None).unwrap();
    }
    let r = stack(
        Axis(1),
        &[right_vector.view(), up_vector.view(), forward_vector.view()],
    )
    .unwrap();
    return r;
}

pub fn orbit_camera(
    mut elevation: f32,
    mut azimuth: f32,
    radius: Option<f32>,
    is_degree: Option<bool>,
    target: Option<ArrayD<f32>>,
    opengl: Option<bool>,
) -> Result<ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>> {
    // # radius: scalar
    // # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    // # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    // # return: [4, 4], camera pose matrix
    let radius = radius.unwrap_or(1.0);
    let is_degree = is_degree.unwrap_or(true);
    let target = target.unwrap_or(ArrayD::zeros(vec![3]));
    let opengl = opengl.unwrap_or(true);
    if is_degree {
        elevation = elevation.to_radians();
        azimuth = azimuth.to_radians();
    }
    let x = radius * elevation.cos() * azimuth.sin();
    let y = -radius * elevation.sin();
    let z = radius * elevation.cos() * azimuth.cos();
    let campos = &array!([x, y, z]) + &target;
    let mut t = Array::eye(4);

    // T[:3, :3] = look_at(campos, target, opengl)
    // T[:3, 3] = campos

    let look_at_matrix = look_at(campos.clone(), target, Some(opengl));
    t.slice_mut(s![..3, ..3]).assign(&look_at_matrix);
    t.slice_mut(s![..3, 3]).assign(&campos);
    return Ok(t);
}

pub struct OrbitCamera {
    w: i32,
    h: i32,
    radius: f32,
    fovy: f32,
    near: f32,
    far: f32,
    center: Array2<f32>,
    rot: Rotation3<f32>,
    up: Array2<f32>,
}

impl OrbitCamera {
    pub fn new(w: i32, h: i32, radius: f32, fovy: f32, near: f32, far: f32) -> Self {
        OrbitCamera {
            w,
            h,
            radius,
            fovy: fovy.to_radians(),
            near,
            far,
            center: array!([0.0, 0.0, 0.0]),
            rot: Rotation3::from_matrix_unchecked(Matrix3::<f32>::identity()),
            up: array!([0.0, 1.0, 0.0]),
        }
    }

    pub fn fovx(&self) -> f32 {
        2.0 * ((self.fovy / 2.0).tan() * self.w as f32 / self.h as f32).atan()
    }

    pub fn campos(&self) -> ArrayD<f32> {
        let pose_matrix = self.pose();
        let campos_slice = pose_matrix.slice(s![..3, 3]);
        let campos_array = ArrayD::from_shape_vec(IxDyn(&[3]), campos_slice.to_vec()).unwrap();
        campos_array
    }

    pub fn pose(&self) -> Array2<f32> {
        let mut res = Array::eye(4); // Create a 4x4 identity matrix as a dynamic array
        res[[2, 3]] = self.radius; // Set the translation part for the z-axis

        // Convert Rotation3<f32> to a 3x3 ndarray and embed it into a 4x4 ndarray matrix
        let rot_matrix = self.rot.to_homogeneous(); // This is a 4x4 matrix in nalgebra
        let rot_ndarray: Array2<f32> =
            Array::from_shape_vec((4, 4), rot_matrix.iter().cloned().collect()).unwrap();

        // Matrix multiplication using `dot` from ndarray
        res = rot_ndarray.dot(&res);

        // Assign the center to the translation part of the matrix
        if self.center.ndim() == 1 && self.center.shape()[0] == 3 {
            res.slice_mut(s![..3, 3]).assign(&self.center);
        }

        res
    }

    pub fn view(&self) -> Array2<f32> {
        self.pose().inv().unwrap()
    }

    pub fn perspective(&self) -> Array2<f32> {
        let y = (self.fovy.to_radians() / 2.0).tan();
        let aspect = self.w as f32 / self.h as f32;
        let perspective_matrix = arr2(&[
            [1.0 / (y * aspect), 0.0, 0.0, 0.0],
            [0.0, -1.0 / y, 0.0, 0.0],
            [
                0.0,
                0.0,
                -(self.far + self.near) / (self.far - self.near),
                -(2.0 * self.far * self.near) / (self.far - self.near),
            ],
            [0.0, 0.0, -1.0, 0.0],
        ]);
        perspective_matrix
    }

    pub fn intrinsics(&self) -> [f32; 4] {
        let focal = self.h as f32 / (2.0 * (self.fovy / 2.0).tan());
        [focal, focal, self.w as f32 / 2.0, self.h as f32 / 2.0]
    }

    pub fn mvp(&self) -> Array2<f32> {
        let perspective = self.perspective();
        let pose = self.pose();
        let pose_inv = pose.inv().expect("Pose matrix is not invertible");

        // Perform matrix multiplication using `dot`
        let mvp_matrix = perspective.dot(&pose_inv);
        mvp_matrix
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        // Assuming self.up is an Array2<f32> and behaves like a 1D vector
        let up_nalgebra = Vector3::new(self.up[[0, 0]], self.up[[0, 1]], self.up[[0, 2]]);

        // Extract the side vector from the rotation matrix
        let side = self.rot.matrix().column(0).into_owned();

        // Calculate rotation vectors based on input deltas
        let rotvec_x = up_nalgebra * (-0.05 * dx).to_radians();
        let rotvec_y = side * (-0.05 * dy).to_radians();

        // Normalize the rotation vectors to create unit vectors
        let unit_rotvec_x = Unit::new_normalize(rotvec_x);
        let unit_rotvec_y = Unit::new_normalize(rotvec_y);

        // Create rotations from rotation vectors
        let rot_x = Rotation3::from_axis_angle(&unit_rotvec_x, unit_rotvec_x.magnitude());
        let rot_y = Rotation3::from_axis_angle(&unit_rotvec_y, unit_rotvec_y.magnitude());

        // Update self.rot by combining new rotations with the existing one
        self.rot = rot_x * rot_y * self.rot;
    }

    pub fn scale(&mut self, delta: f32) {
        self.radius *= 1.1f32.powf(-delta);
    }

    pub fn pan(&mut self, dx: f32, dy: f32, dz: f32) {
        let delta = Vector3::new(-dx, -dy, dz);

        // Scale the delta by the sensitivity factor
        let scaled_delta = 0.0005 * delta;

        // Apply the rotation to the delta vector
        let rotated_delta = self.rot * scaled_delta;

        // Convert rotated_delta back to ndarray to perform addition
        let rotated_delta_ndarray =
            Array1::from_vec(vec![rotated_delta.x, rotated_delta.y, rotated_delta.z]);
        self.center += &rotated_delta_ndarray;
    }
}
