mod array_utils;
mod cam_utils;
mod grid_put;

mod cli;
mod gs_renderer;
mod mesh;
mod mesh_utils;
mod ply_mesh;
mod sh_utils;

fn main() {
    let cli::MainParams {
        config,
        mesh,
        input,
        save_path,
        ..
    } = cli::get_main_params();

    println!("Config: {}", config);
    println!("Input: {}", input);
    println!("Save path: {}", save_path);
    println!("Hello from main!");
}
