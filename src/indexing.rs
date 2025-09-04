use std::{collections::HashMap, path::PathBuf, sync::Arc};

use bytes::Bytes;
use chrono_tz::Europe::Paris;
use meilisearch_sdk::{client::Client, indexes::Index};
use ndarray::Array4;
use image::{self, imageops::FilterType, DynamicImage};
use ort::{inputs, session::{builder::GraphOptimizationLevel, Session}};
use serde::Serialize;
use tokio::{fs, sync::RwLock};
use log::info;

#[derive(Clone)]
pub struct AppState {
    session: Arc<RwLock<Session>>,
    faces: Index,
    imgs_path: PathBuf,
}

#[derive(Serialize)]
struct Face {
    id: u32,
    name: Option<String>,
    #[serde(rename = "_vectors")]
    vectors: HashMap<String, Option<Vec<f32>>>,
    source_url: Option<String>,
    date: Option<String>,
}

pub async fn app_state(meili_url: &str, meili_key: &str, index: &str, imgs_path: impl Into<PathBuf>) -> Result<AppState, Box<dyn std::error::Error>> {
    let session = load_model().await?;
    let key_opt: Option<String> = if meili_key.is_empty() { None } else { Some(meili_key.to_string()) };
    let client = Client::new(meili_url, key_opt)?;
    let faces = client.index(index);
    Ok(AppState {
        session: Arc::new(RwLock::new(session)),
        faces,
        imgs_path: imgs_path.into(),
    })
}

async fn load_model() -> Result<Session, Box<dyn std::error::Error>> {
    // Download & cache the model locally if missing
    let model_url = "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx";
    let model_path = "model.onnx";
    if fs::metadata(model_path).await.is_err() {
        let client = reqwest::Client::new();
        let model_bytes = client.get(model_url).send().await?.bytes().await?;
        fs::write(model_path, &model_bytes).await?;
    }

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(6)?
        .commit_from_file(model_path)?;
    
    info!("Model loaded successfully {model:?}");
    Ok(model)
}

fn preprocess_nhwc(img: &DynamicImage) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    // Resize quickly, convert to RGB8, then normalize to f32 NHWC: (x-127.5)/128
    // Triangle is a good balance of speed/quality vs Lanczos3.
    let img = img.to_rgb8();
    let (w, h) = img.dimensions(); // (112, 112)
    let raw = img.as_raw(); // &[u8] in RGBRGB...
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

pub async fn index(bytes: Bytes, app_state: Arc<AppState>, source_url: Option<String>, name: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    match image::load_from_memory(&bytes) {
        Ok(img) => {
            let img = img.resize_exact(112, 112, FilterType::Triangle);
            let input = preprocess_nhwc(&img)?;
            let input_value = ort::value::Value::from_array(input)?;
            let mut model = app_state.session.write().await;
            let outputs = model.run(inputs!["input_1" => input_value])?;
            let predictions = outputs["embedding"].try_extract_array::<f32>()?;
            // Log only a small slice to avoid large string formatting overhead
            let flat = predictions.as_slice().unwrap_or(&[]);
            let preview_len = flat.len().min(5);

            // Prepare vectors payload expected by Meilisearch
            let mut vectors = HashMap::new();
            vectors.insert("embedding".to_string(), Some(flat.to_vec()));
            vectors.insert("default".to_string(), Some(flat.to_vec()));
            let id = rand::random::<u32>();
            let img_path = app_state.imgs_path.to_str().unwrap_or("./images");
            img.save(format!("{img_path}/{id}.jpg"))?;
            let _ = app_state.faces.add_documents(&[Face {
                id,
                name,
                vectors,
                source_url,
                date: Some(chrono::Utc::now().with_timezone(&Paris).to_rfc3339()),
            }], Some("id")).await?;
            info!("Embedding preview: {:?} ({} dims)", &flat[..preview_len], flat.len());
        }
        Err(e) => {
            eprintln!("Error decoding image: {e}");
        }
    }
    Ok(())
}
