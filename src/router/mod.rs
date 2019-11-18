use crate::rocket::State;
use crate::{Arc, Bbox, Json, Receiver, Sender, NAMES};

#[derive(Debug, Clone)]
pub enum RequestType {
    BBOXES,
    IMAGE,
}

#[derive(Debug, Clone)]
pub struct ImageRequestVectorized {
    pub img: Vec<u8>,
    pub request: RequestType,
}

#[derive(Responder)]
pub enum RouterResponse {
    #[response(status = 200, content_type = "binary")]
    ImageBuffer(Vec<u8>),
    #[response(status = 200, content_type = "application/json")]
    OkBboxes(Json<Vec<Vec<Bbox>>>),
    #[response(status = 200, content_type = "application/json")]
    OkNames(Json<Vec<String>>),
    #[response(status = 400)]
    NotFound(String),
    #[response(status = 500, content_type = "plain")]
    Error(String),
}

#[get("/")]
pub fn get() -> RouterResponse {
    RouterResponse::NotFound(format!("Not Found"))
}

#[get("/boxes_names", format = "json")]
pub fn get_names() -> RouterResponse {
    let mut names = Vec::new();
    NAMES.iter().for_each(|name| {
        names.push(format!("{}", name));
    });
    RouterResponse::OkNames(Json(names))
}

#[post("/boxes", format = "binary", data = "<data>")]
pub fn post_recognize_objects_boxes(
    data: Vec<u8>,
    sender: State<Arc<Sender<ImageRequestVectorized>>>,
    receiver: State<Arc<Receiver<Vec<Vec<Bbox>>>>>,
) -> RouterResponse {
    info!("Received image buffer of length: {:?} bytes", &data.len());
    let img_vectorized = ImageRequestVectorized {
        img: data,
        request: RequestType::BBOXES,
    };
    if let Ok(_) = sender.send(img_vectorized) {
        if let Ok(solution) = receiver.recv() {
            return RouterResponse::OkBboxes(Json(solution));
        }
    }
    RouterResponse::Error(format!("Error: Cannot handle operation."))
}

#[post("/image", format = "binary", data = "<data>")]
pub fn post_recognize_objects_images(
    data: Vec<u8>,
    sender: State<Arc<Sender<ImageRequestVectorized>>>,
    receiver: State<Arc<Receiver<Vec<u8>>>>,
) -> RouterResponse {
    info!("Received image buffer of length: {:?} bytes", &data.len());
    let img_vectorized = ImageRequestVectorized {
        img: data,
        request: RequestType::IMAGE,
    };
    if let Ok(_) = sender.send(img_vectorized) {
        if let Ok(solution_img_buffer) = receiver.recv() {
            return RouterResponse::ImageBuffer(solution_img_buffer);
        }
    }
    RouterResponse::Error(format!("Error: Cannot handle operation."))
}
