use std::io::Read;

use actix_multipart::form::{tempfile::TempFile, text::Text, MultipartForm};
use actix_web::{get, post, web::Data, HttpResponse, Responder};

use crate::indexing::{index, AppState};

#[derive(Debug, MultipartForm)]
struct Form {
    #[multipart(limit = "1MB")]
    image: TempFile,
    name: Option<Text<String>>,
    source_url: Option<Text<String>>,
    date: Option<Text<String>>,
    api_key: Text<String>,
}


#[post("/upload")]
async fn upload(app_data: Data<AppState>, MultipartForm(form): MultipartForm<Form>) -> impl Responder {
    let mut buf = Vec::new();
    form.image.file.as_file().read_to_end(&mut buf).unwrap();
    let source_url = form.source_url.map(|t| t.into_inner());
    let name = form.name.map(|t| t.into_inner());
    match index(bytes::Bytes::from(buf), app_data.into_inner(), source_url, name).await {
        Ok(_) => HttpResponse::Ok().body(""),
        Err(e) => HttpResponse::InternalServerError().body(format!("Error: {e}")),
    }
}

#[get("/search")]
async fn search(MultipartForm(form): MultipartForm<Form>) -> impl Responder {
    HttpResponse::Ok().body("Search endpoint")
}
