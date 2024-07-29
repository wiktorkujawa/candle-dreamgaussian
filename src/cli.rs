use clap::{Arg, Command};

pub struct MainParams {
    pub config: String,
    pub mesh: String,
    pub input: String,
    pub save_path: String,
    pub gui: bool,
}

pub struct ProcessParams {
    pub path: String,
    pub size: usize,
    pub model: String,
    pub border_ratio: f64,
    pub recenter: bool,
}

// Function to parse CLI arguments and return MainParams
pub fn get_main_params() -> MainParams {
    let matches = Command::new("DreamGaussian")
        .arg(
            Arg::new("config")
                .long("config")
                .required(true)
                .help("Path to the configuration file"),
        )
        .arg(
            Arg::new("input")
                .long("input")
                .default_value("data/scientist.png")
                .help("Input image file"),
        )
        .arg(
            Arg::new("mesh")
                .long("mesh")
                .default_value("logs/scientist.obj")
                .help("Input mesh file"),
        )
        .arg(
            Arg::new("save_path")
                .long("save_path")
                .default_value("output")
                .help("Path to save the output"),
        )
        .arg(
            Arg::new("gui")
                .long("gui")
                .help("Enable GUI mode")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    MainParams {
        config: matches.get_one::<String>("config").unwrap().clone(),
        mesh: matches.get_one::<String>("mesh").unwrap().clone(),
        input: matches.get_one::<String>("input").unwrap().clone(),
        save_path: matches.get_one::<String>("save_path").unwrap().clone(),
        gui: matches.get_flag("gui"),
    }
}

pub fn get_process_params() -> ProcessParams {
    let matches = Command::new("Process")
        .arg(
            Arg::new("path")
                .required(true)
                .help("Path to the directory or image file"),
        )
        .arg(
            Arg::new("size")
                .long("size")
                .default_value("256")
                .help("Size of the output"),
        )
        .arg(
            Arg::new("model")
                .long("model")
                .default_value("model.onnx")
                .help("rembg model, see https://huggingface.co/models"),
        )
        .arg(
            Arg::new("border_ratio")
                .long("border_ratio")
                .default_value("0.2")
                .help("Output border ratio"),
        )
        .arg(
            Arg::new("recenter")
                .long("recenter")
                .default_value("true")
                .help("Recenter, potentially not helpful for multiview zero123")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    ProcessParams {
        path: matches.get_one::<String>("path").unwrap().clone(),
        size: matches.get_one::<String>("size").unwrap().parse().unwrap(),
        model: matches.get_one::<String>("model").unwrap().clone(),
        border_ratio: matches
            .get_one::<String>("border_ratio")
            .unwrap()
            .parse::<f64>()
            .unwrap(),
        recenter: matches.get_flag("recenter"),
    }
}
