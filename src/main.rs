use std::{collections::HashMap, sync::Arc};

use actix_web::{web, App, HttpServer};
use meilisearch_sdk::{settings::{EmbedderSource, Embedder}, client::Client};
use tokio::fs;
use dotenv::dotenv;
use tokio::time::{sleep, Duration};

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
    log::info!("facedb starting up");

    // Ensure the images directory exists
    let imgs_path = std::env::var("IMG_FOLDER").unwrap_or_else(|_| "./images".to_string());
    fs::create_dir_all(&imgs_path).await?;
    log::info!("Ensured image folder exists at {imgs_path}");
    
    // Prepare meilisearch index
    let meili_host = std::env::var("MEILI_HOST").unwrap_or_else(|_| "http://localhost:7700".to_string());
    let meili_key_opt = std::env::var("MEILI_MASTER_KEY").ok();
    log::info!("Connecting to Meilisearch at {meili_host}");
    let client = Client::new(&meili_host, meili_key_opt.as_deref()).unwrap();
    let faces = client.index("faces");
    // Retry setting embedders until Meilisearch is ready
    let embedders = HashMap::from([
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
    ]);
    for attempt in 1..=30u8 {
        match faces.set_embedders(&embedders).await {
            Ok(_) => break,
            Err(e) => {
                log::warn!("Meilisearch not ready for embedders (attempt {attempt}/30): {e}");
                sleep(Duration::from_secs(1)).await;
            }
        }
    }

    // App state
    let app_state = Arc::new(indexing::app_state(
        &meili_host,
    meili_key_opt.as_deref().unwrap_or(""),
        "faces",
        imgs_path,
    ).await?);

    let app_state_clone = app_state.clone();
    tokio::spawn(async move {
        let _ = scraper::scrap(app_state_clone).await;
    });
    log::info!("Scraper task spawned");

    log::info!("Starting HTTP server on 0.0.0.0:8080");
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(app_state.clone()))
            .service(server::upload)
            .service(server::search)
    })
    // Bind to 0.0.0.0 so the service is reachable from outside the container
    .bind(("0.0.0.0", 8080))?
    .run()
    .await?;

    log::info!("HTTP server stopped; shutting down facedb");

    Ok(())
}
