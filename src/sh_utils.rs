// #  Copyright 2021 The PlenOctree Authors.
// #  Redistribution and use in source and binary forms, with or without
// #  modification, are permitted provided that the following conditions are met:
// #
// #  1. Redistributions of source code must retain the above copyright notice,
// #  this list of conditions and the following disclaimer.
// #
// #  2. Redistributions in binary form must reproduce the above copyright notice,
// #  this list of conditions and the following disclaimer in the documentation
// #  and/or other materials provided with the distribution.
// #
// #  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// #  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// #  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// #  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// #  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// #  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// #  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// #  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// #  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// #  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// #  POSSIBILITY OF SUCH DAMAGE.
use candle_core::{IndexOp, Result, Tensor};

const C0: f32 = 0.28209479177387814;
const C1: f32 = 0.4886025119029199;
const C2: [f32; 5] = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
];
const C3: [f32; 7] = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
];
const C4: [f32; 9] = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
];

pub fn eval_sh(deg: i32, sh: &Tensor, dirs: &Tensor) -> Result<Tensor> {
    assert!(deg <= 4 && deg >= 0);
    let coeff = (deg + 1).pow(2);
    assert!(sh.dims()[sh.dims().len() - 1] as i32 >= coeff);


    // let c0_tensor = Tensor::full(C0, sh.dim(D::Minus1)?, sh.device())?;
    // let c1_tensor = Tensor::full(C1, sh.dim(D::Minus1)?, sh.device())?;

    // let c0_tensor = Tensor::full(C0, &[1], sh.device())?;
    // let c1_tensor = Tensor::full(C1, &[1], sh.device())?;

    let c0_tensor = Tensor::new(C0, sh.device())?;
    let c1_tensor = Tensor::new(C1, sh.device())?;

    let mut result = (c0_tensor * sh.i((.., 0))?)?;
    if deg > 0 {
        let x = dirs.i((.., 0..1))?;
        let y = dirs.i((.., 1..2))?;
        let z = dirs.i((.., 2..3))?;

        result = (result - (&y * &c1_tensor * &sh.i((.., 1))?)?
            + (&z * &c1_tensor * &sh.i((.., 2))?)?
            - (&x * &c1_tensor * &sh.i((.., 3))?)?)?;

        // Continue with higher degrees...
    }

    Ok(result)
}

pub fn rgb_to_sh(rgb: &Tensor) -> Result<Tensor> {

    let c0_tensor = Tensor::new(C0, rgb.device())?;

    Ok(((rgb - 0.5)? / c0_tensor)?)
}

pub fn sh_to_rgb(sh: &Tensor) -> Result<Tensor> {
    let c0_tensor = Tensor::new(C0, sh.device())?;

    Ok(((sh * c0_tensor)? + 0.5)?)
}
