use actix_multipart::form::{tempfile::TempFile, text::Text, MultipartForm};
use actix_web::{get, post, HttpResponse, Responder};

#[derive(Debug, MultipartForm)]
struct Form {
    #[multipart(limit = "1MB")]
    image: TempFile,
    name: Option<Text<String>>,
    source_url: Option<Text<String>>,
    date: Option<Text<String>>,
}


#[post("/upload")]
async fn upload(MultipartForm(form): MultipartForm<Form>) -> impl Responder {

    HttpResponse::Ok().body("Hello world!")
}

#[get("/search")]
async fn search(MultipartForm(form): MultipartForm<Form>) -> impl Responder {
    HttpResponse::Ok().body("Search endpoint")
}
