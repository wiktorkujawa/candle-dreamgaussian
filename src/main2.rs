mod cli;

fn main() {
    let params = cli::get_main_params();

    println!("Config: {}", params.config);
    println!("Input: {}", params.input);
    println!("Save path: {}", params.save_path);
    println!("GUI mode: {}", params.gui);

    println!("Hello from main2!");
}
