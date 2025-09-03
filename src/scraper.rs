use std::sync::Arc;
use tokio::time;
use crate::{fetch_image, indexing::{index, AppState}};

pub async fn scrap(app_state: Arc<AppState>) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::builder()
        .pool_idle_timeout(std::time::Duration::from_secs(30))
        .tcp_nodelay(true)
        .build()?;

    let mut ticker = time::interval(time::Duration::from_millis(995));
    loop {
        ticker.tick().await;

        match fetch_image(&client, "https://thispersondoesnotexist.com/").await {
            Ok(image_data) => {
                index(image_data, app_state.clone()).await?;
            }
            Err(e) => {
                eprintln!("Error fetching image: {e}");
            }
        }
    }
}
