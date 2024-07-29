use image::{imageops, open, DynamicImage, GenericImageView, RgbaImage};
use rmbg::Rmbg;
use std::{fs, path::Path};

mod cli;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let process_params = cli::get_process_params();

    let path = &process_params.path;
    let size = process_params.size as i64;
    let model = &process_params.model;
    let _border_ratio = process_params.border_ratio;
    let recenter = process_params.recenter;

    let rmbg = Rmbg::new(model).expect("Failed to load model");

    let paths = if Path::new(path).is_dir() {
        fs::read_dir(path)?
            .map(|res| res.map(|e| e.path()))
            .collect::<Result<Vec<_>, std::io::Error>>()?
    } else {
        vec![Path::new(path).to_path_buf()]
    };

    for path in paths {
        let file_name = path.file_stem().unwrap().to_str().unwrap();
        let out_rgba = format!("transformed/{}_rgba.png", file_name);

        let original_img = open(path.to_str().unwrap()).expect("Failed to open image");
        let carved_image = rmbg
            .remove_background(&original_img)
            .expect("Failed to remove background");

        let final_image = if recenter {
            // Calculate the bounding box of non-transparent pixels
            let mut min_x = carved_image.width();
            let mut max_x = 0;
            let mut min_y = carved_image.height();
            let mut max_y = 0;

            for (x, y, pixel) in carved_image.pixels() {
                if pixel[3] > 0 {
                    // Check alpha channel
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }

            let width = max_x - min_x + 1;
            let height = max_y - min_y + 1;

            // Calculate new dimensions based on border ratio
            let desired_size = (size as f64 * (1.0 - _border_ratio)) as u32;
            let scale = desired_size as f32 / width.max(height) as f32;
            let new_width = (width as f32 * scale) as u32;
            let new_height = (height as f32 * scale) as u32;

            // Resize the cropped region
            let cropped = carved_image.crop_imm(min_x, min_y, width, height);
            let resized = cropped.resize(
                new_width,
                new_height,
                image::imageops::FilterType::CatmullRom,
            );

            // Create a new image and center the resized image
            let mut final_image = RgbaImage::new(size as u32, size as u32);
            let x_offset = (size - new_width as i64) / 2;
            let y_offset = (size - new_height as i64) / 2;
            imageops::overlay(&mut final_image, &resized, x_offset, y_offset);

            DynamicImage::ImageRgba8(final_image)
        } else {
            carved_image
        };

        final_image.save(out_rgba).expect("Failed to save image");
    }

    Ok(())
}
