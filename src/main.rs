use ndarray::Array4;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use tokio::fs;
use tokio::time;
use image::{self, imageops::FilterType, DynamicImage};
use log::info;

pub async fn fetch_image(client: &reqwest::Client, url: &str) -> Result<bytes::Bytes, reqwest::Error> {
    let response = client.get(url).send().await?;
    let bytes = response.bytes().await?;
    Ok(bytes)
}

fn preprocess_nhwc(img: DynamicImage) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    // Resize quickly, convert to RGB8, then normalize to f32 NHWC: (x-127.5)/128
    // Triangle is a good balance of speed/quality vs Lanczos3.
    let rgb = img.resize_exact(112, 112, FilterType::Triangle).to_rgb8();

    let (w, h) = rgb.dimensions(); // (112, 112)
    let raw = rgb.as_raw(); // &[u8] in RGBRGB...
    let numel = (h as usize) * (w as usize) * 3;
    let mut data = vec![0f32; numel];
    for (i, px) in raw.chunks_exact(3).enumerate() {
        let base = i * 3;
        data[base] = (px[0] as f32 - 127.5) / 128.0; // R
        data[base + 1] = (px[1] as f32 - 127.5) / 128.0; // G
        data[base + 2] = (px[2] as f32 - 127.5) / 128.0; // B
    }

    // Build the array without an extra zero-initialization pass.
    let arr = Array4::from_shape_vec((1, h as usize, w as usize, 3), data)?;
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
    // Reuse HTTP client for connection pooling
    let client = reqwest::Client::builder()
        .pool_idle_timeout(std::time::Duration::from_secs(30))
        .tcp_nodelay(true)
        .build()?;

    // Download & cache the model locally if missing
    let model_url = "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx";
    let model_path = "model.onnx";
    if fs::metadata(model_path).await.is_err() {
        let model_bytes = client.get(model_url).send().await?.bytes().await?;
        fs::write(model_path, &model_bytes).await?;
    }
    let mut model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(6)?
        .commit_from_file(model_path)?;
    info!("Model loaded successfully {model:?}");

    loop {
        ticker.tick().await;

    match fetch_image(&client, "https://thispersondoesnotexist.com/").await {
            Ok(image_data) => {
                match image::load_from_memory(&image_data) {
                    Ok(decoded) => {
                        let input = preprocess_nhwc(decoded)?;
                        let input_value = ort::value::Value::from_array(input)?;
                        let outputs = model.run(inputs!["input_1" => input_value])?;
                        let predictions = outputs["embedding"].try_extract_array::<f32>()?;
                        // Log only a small slice to avoid large string formatting overhead
                        let flat = predictions.as_slice().unwrap_or(&[]);
                        let preview_len = flat.len().min(5);
                        info!("Embedding preview: {:?} ({} dims)", &flat[..preview_len], flat.len());
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
