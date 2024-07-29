use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

pub struct Mesh {
    v: Tensor,
    f: Tensor,
    vn: Option<Tensor>,
    vt: Option<Tensor>,
    albedo: Option<Tensor>,
    device: Device,
}

impl Mesh {
    pub fn new(device: Device) -> Self {
        let v = Tensor::zeros(&[0, 3], DType::F32, &device).unwrap();
        let f = Tensor::zeros(&[0, 3], DType::U32, &device).unwrap();
        Self {
            v,
            f,
            vn: None,
            vt: None,
            albedo: None,
            device,
        }
    }

    pub fn load_obj(path: &Path, device: &Device) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut texcoords = Vec::new();
        let mut normals = Vec::new();
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            match parts[0] {
                "v" => {
                    let coords: Vec<f32> = parts[1..]
                        .iter()
                        .map(|x| x.parse::<f32>().unwrap())
                        .collect();
                    vertices.extend(coords);
                }
                "vt" => {
                    let coords: Vec<f32> = parts[1..]
                        .iter()
                        .map(|x| x.parse::<f32>().unwrap())
                        .collect();
                    texcoords.extend(coords);
                }
                "vn" => {
                    let coords: Vec<f32> = parts[1..]
                        .iter()
                        .map(|x| x.parse::<f32>().unwrap())
                        .collect();
                    normals.extend(coords);
                }
                "f" => {
                    let indices: Vec<u32> = parts[1..]
                        .iter()
                        .map(|x| x.split('/').next().unwrap().parse::<u32>().unwrap() - 1)
                        .collect();
                    faces.extend(indices);
                }
                _ => {}
            }
        }

        let v = Tensor::from_vec(vertices.clone(), &[vertices.len() / 3, 3], device)?;
        let f = Tensor::from_vec(faces.clone(), &[faces.len() / 3, 3], device)?;

        Ok(Self {
            v,
            f,
            vn: None,
            vt: None,
            albedo: None,
            device: device.clone(),
        })
    }
    /// Computes vertex normals for a mesh using the available tensor operations.
    pub fn auto_normal(&mut self) -> Result<&mut Self> {
        let i0 = self.f.i((.., 0))?.to_dtype(DType::I64)?;
        let i1 = self.f.i((.., 1))?.to_dtype(DType::I64)?;
        let i2 = self.f.i((.., 2))?.to_dtype(DType::I64)?;

        let v0 = self.v.i(&i0)?;
        let v1 = self.v.i(&i1)?;
        let v2 = self.v.i(&i2)?;

        // let face_normals = (v1.sub(&v0)?).cross(&(v2.sub(&v0)?))?;
        // Manually compute the cross product
        let a1 = v1.sub(&v0)?;
        let a2 = v2.sub(&v0)?;
        let cross_x = a1
            .i((.., 1))?
            .mul(&a2.i((.., 2))?)?
            .sub(&a1.i((.., 2))?.mul(&a2.i((.., 1))?)?)?;
        let cross_y = a1
            .i((.., 2))?
            .mul(&a2.i((.., 0))?)?
            .sub(&a1.i((.., 0))?.mul(&a2.i((.., 2))?)?)?;
        let cross_z = a1
            .i((.., 0))?
            .mul(&a2.i((.., 1))?)?
            .sub(&a1.i((.., 1))?.mul(&a2.i((.., 0))?)?)?;
        let face_normals = Tensor::stack(&[cross_x, cross_y, cross_z], 1)?;

        // Splat face normals to vertices
        let mut vn = Tensor::zeros_like(&self.v)?;
        vn = vn.scatter_add(&i0.unsqueeze(1)?.repeat(&[1, 3])?, &face_normals, 0)?;
        vn = vn.scatter_add(&i1.unsqueeze(1)?.repeat(&[1, 3])?, &face_normals, 0)?;
        vn = vn.scatter_add(&i2.unsqueeze(1)?.repeat(&[1, 3])?, &face_normals, 0)?;

        // Manually compute the dot product for each vector in vn
        let vn_x = vn.i((.., 0))?;
        let vn_y = vn.i((.., 1))?;
        let vn_z = vn.i((.., 2))?;

        let magnitude_squared = vn_x
            .mul(&vn_x)?
            .add(&vn_y.mul(&vn_y)?)?
            .add(&vn_z.mul(&vn_z)?)?;
        let magnitude = magnitude_squared.sqrt()?;
        let condition = magnitude_squared.gt(1e-20)?;

        let normalized_vn_x = vn_x.div(&magnitude)?;
        let normalized_vn_y = vn_y.div(&magnitude)?;
        let normalized_vn_z = vn_z.div(&magnitude)?;
        let normalized_vn = Tensor::stack(&[normalized_vn_x, normalized_vn_y, normalized_vn_z], 1)?;

        let default_normal =
            Tensor::new(&[0.0, 0.0, 1.0], &self.v.device())?.to_dtype(DType::F32)?;
        vn = condition.where_cond(&normalized_vn, &default_normal)?;

        self.vn = Some(vn);
        Ok(self)
    }

    pub fn write_obj(&self, path: &Path) -> Result<()> {
        let mut file = BufWriter::new(File::create(path)?);
        let mtl_path = path.with_extension("mtl");
        let albedo_path = path.with_file_name("albedo.png");

        let v_size = self.v.shape().dims()[0];
        let f_size = self.f.shape().dims()[0];

        // Write MTL file reference at the top of the OBJ file
        writeln!(
            file,
            "mtllib {}",
            mtl_path.file_name().unwrap().to_str().unwrap()
        )?;

        // Write vertices
        for i in 0..self.v.shape().dims()[0] {
            let vertex = self.v.i(i)?;
            writeln!(file, "v {} {} {}", vertex.i(0)?, vertex.i(1)?, vertex.i(2)?)?;
        }

        // Write texture coordinates if available
        if let Some(vt) = &self.vt {
            for i in 0..vt.shape().dims()[0] {
                let tex_coord = vt.i(i)?;
                writeln!(file, "vt {} {}", tex_coord.i(0)?, tex_coord.i(1)?)?;
            }
        }

        // Write normals if available
        if let Some(vn) = &self.vn {
            for i in 0..vn.shape().dims()[0] {
                let normal = vn.i(i)?;
                writeln!(
                    file,
                    "vn {} {} {}",
                    normal.i(0)?,
                    normal.i(1)?,
                    normal.i(2)?
                )?;
            }
        }

        // Write faces
        writeln!(file, "usemtl defaultMat")?;
        for i in 0..f_size {
            let face = self.f.i(i)?;
            // Assuming 1-based index for OBJ files
            writeln!(
                file,
                "f {}/{}/{} {}/{}/{} {}/{}/{}",
                //  face.i(0)? + 1, face.i(0)? + 1, face.i(0)? + 1,
                //  face.i(1)? + 1, face.i(1)? + 1, face.i(1)? + 1,
                //  face.i(2)? + 1, face.i(2)? + 1, face.i(2)? + 1
                face.i(0)?,
                face.i(0)?,
                face.i(0)?,
                face.i(1)?,
                face.i(1)?,
                face.i(1)?,
                face.i(2)?,
                face.i(2)?,
                face.i(2)?
            )?;
        }

        // Write MTL file
        let mut mtl_file = BufWriter::new(File::create(mtl_path)?);
        writeln!(mtl_file, "newmtl defaultMat")?;
        writeln!(mtl_file, "Ka 1 1 1")?;
        writeln!(mtl_file, "Kd 1 1 1")?;
        writeln!(mtl_file, "Ks 0 0 0")?;
        writeln!(mtl_file, "Tr 1")?;
        writeln!(mtl_file, "illum 1")?;
        writeln!(mtl_file, "Ns 0")?;
        writeln!(
            mtl_file,
            "map_Kd {}",
            albedo_path.file_name().unwrap().to_str().unwrap()
        )?;

        Ok(())
    }
}
