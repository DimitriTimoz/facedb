use ndarray::Array4;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use reqwest;
use tokio::time;
use tokio::fs;
use image::{self, imageops::FilterType, DynamicImage};
use log::{info, error};
use std::convert::TryInto;

pub async fn fetch_image(url: &str) -> Result<bytes::Bytes, reqwest::Error> {
    let response = reqwest::get(url).await?;
    let bytes = response.bytes().await?;
    Ok(bytes)
}

fn preprocess_nhwc(rgb: DynamicImage) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    // 112x112, RGB, f32, NHWC, (x-127.5)/128
    let rgb = rgb.resize_exact(112, 112, FilterType::Lanczos3).to_rgb8();
    rgb.save("resized_image.png")?;

    let (w, h) = rgb.dimensions(); // (112, 112)
    let mut arr = Array4::<f32>::zeros((1, h as usize, w as usize, 3));
    for y in 0..h as usize {
        for x in 0..w as usize {
            let p = rgb.get_pixel(x as u32, y as u32);
            // RGB order
            let r = (p[0] as f32 - 127.5) / 128.0;
            let g = (p[1] as f32 - 127.5) / 128.0;
            let b = (p[2] as f32 - 127.5) / 128.0;
            arr[(0, y, x, 0)] = r;
            arr[(0, y, x, 1)] = g;
            arr[(0, y, x, 2)] = b;
        }
    }
    Ok(arr)
}

fn l2_normalize(v: &mut [f32]) {
    let mut norm = 0.0f32;
    for x in v.iter() { norm += x * x; }
    norm = norm.sqrt().max(1e-12);
    for x in v.iter_mut() { *x /= norm; }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let mut ticker = time::interval(time::Duration::from_millis(995));
    // Download & save the model
    let model_url = "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx";
    let model_bytes = reqwest::get(model_url).await?.bytes().await?;
    let model_path = "model.onnx";
    fs::write(model_path, &model_bytes).await?;
    let mut model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;
    info!("Model loaded successfully {model:?}");

    loop {
        ticker.tick().await;

        match fetch_image("https://thispersondoesnotexist.com/").await {
            Ok(image_data) => {
                match image::load_from_memory(&image_data) {
                    Ok(decoded) => {
                        let input = preprocess_nhwc(decoded)?;
                        let input_value = ort::value::Value::from_array(input)?;
                        let outputs = model.run(inputs!["input_1" => input_value])?;
                        let predictions = outputs["embedding"].try_extract_array::<f32>()?;

                        info!("Output {predictions:?}");
                    }
                    Err(e) => {
                        eprintln!("Error decoding image: {e}");
                    }
                }
            }
            Err(e) => {
                eprintln!("Error fetching image: {e}");
            }
        }
    }
}
