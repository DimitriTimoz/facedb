use std::sync::Arc;
use log::info;
use tokio::time;
use crate::{fetch_image, indexing::{index, AppState}};

pub async fn scrap(app_state: Arc<AppState>) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::builder()
        .pool_idle_timeout(std::time::Duration::from_secs(30))
        .tcp_nodelay(true)
        .user_agent("Mozilla/5.0 (compatible; facedb-scraper/1.0; +https://example.local)")
        .build()?;

    let mut ticker = time::interval(time::Duration::from_millis(995));
    loop {
        ticker.tick().await;

    // Add a cache buster to avoid cached responses
    let url = format!("https://thispersondoesnotexist.com/?_={}", chrono::Utc::now().timestamp_millis());
    match fetch_image(&client, &url).await {
            Ok(image_data) => {
        index(image_data, app_state.clone(), Some("https://thispersondoesnotexist.com/".to_string()), None).await?;
                info!("Image indexed successfully");
            }
            Err(e) => {
                eprintln!("Error fetching image: {e}");
            }
        }
    }
}
