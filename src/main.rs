use std::{collections::HashMap, sync::Arc};

use meilisearch_sdk::{settings::{EmbedderSource, Embedder}, client::Client};
use tokio::fs;
use dotenv::dotenv;

pub mod server;
pub mod scraper;
pub mod indexing;

pub async fn fetch_image(client: &reqwest::Client, url: &str) -> Result<bytes::Bytes, reqwest::Error> {
    let response = client.get(url).send().await?;
    let bytes = response.bytes().await?;
    Ok(bytes)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    env_logger::init();

    // Ensure the images directory exists
    let imgs_path = std::env::var("IMG_FOLDER").unwrap_or_else(|_| "./images".to_string());
    fs::create_dir_all(&imgs_path).await?;

    // Prepare meilisearch index
    let client = Client::new("http://localhost:7700", Some(std::env::var("MEILI_MASTER_KEY").unwrap())).unwrap();
    let faces = client.index("faces");
    faces.set_embedders(&HashMap::from([
        (
            "embedding".to_string(),
            Embedder {
                source: EmbedderSource::UserProvided,
                dimensions: Some(512),
                ..Embedder::default()
            },
        ),
        (
            "default".to_string(),
            Embedder {
                source: EmbedderSource::UserProvided,
                dimensions: Some(512),
                ..Embedder::default()
            },
        ),
    ])).await?;

    // App state
    let app_state = Arc::new(indexing::app_state(
        "http://localhost:7700",
        &std::env::var("MEILI_MASTER_KEY").unwrap(),
        "faces",
        imgs_path,
    ).await?);

    tokio::spawn(async move {
        let _ = scraper::scrap(app_state).await;
    });

    Ok(())
}
