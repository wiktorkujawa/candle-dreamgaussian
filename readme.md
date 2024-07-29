# Candle-DreamGaussian Sandbox

This repository is created in purpose of testing the features and possibilities of candle library by porting DreamGaussian project[https://dreamgaussian.github.io] to Rust and Candle.

## Installation

Ensure that you have ONNX Runtime (ORT) installed:  
`https://onnxruntime.ai/docs/install/`

and library linked f.e.:
`export ORT_LIB_LOCATION="/usr/local/include/onnxruntime/include"`

More information about installing the library can be found also here:
`https://medium.com/@massimilianoriva96/onnxruntime-integration-with-ubuntu-and-cmake-5d7af482136a`

Before running the application, ensure that all dependencies are installed. You can install the required Rust libraries using Cargo:

cargo build --release

This command also compiles the project, ensuring that all binaries are up to date.



## Usage

### Process Binary - background removal and recentering, save rgba at size x size(default 256 x 256)
Run the `process` binary to remove backgrounds from images or directories of images. Here's the basic command structure:

cargo run --bin process -- directory_or_image_file [options]

#### Parameters:
- `directory_or_image_file`: Path to the image file or directory containing multiple images to process. This parameter is required.

#### Optional Arguments:
- `--size`: Specifies the size to which images should be resized (default is the original size if not specified).
- `--model`: Specifies the model used for background removal (default was downloaded from here: https://huggingface.co/briaai/RMBG-1.4/blob/main/onnx/model.onnx).
- `--border_ratio`: Specifies the border ratio used when recentering images (default is 0.2).
- `--recenter`: Specifies whether to recenter the image (default is false).

#### Example Command:

cargo run --bin process -- "path/to/image_or_directory" --size 300

### Main Binary - training Gaussian Stage
#### Work in progress

### Main2 Binary - training Mesh Stage
#### Work in progress

