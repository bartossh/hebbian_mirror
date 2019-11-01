use crate::rocket::State;
use crate::{decode, Arc, Bbox, Json, Receiver, Sender};

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

#[derive(Debug, Clone, Deserialize)]
pub struct MirrorRequest {
    img: String,
}

#[derive(Responder)]
pub enum RouterResponse {
    #[response(status = 200, content_type = "application/json")]
    OkString(Json<Vec<Vec<Bbox>>>),
    #[response(status = 200, content_type = "binary")]
    OkBinary(Vec<u8>),
    #[response(status = 400)]
    NotFound(String),
    #[response(status = 500, content_type = "plain")]
    Error(String),
}

#[post("/tellmewho", format = "json", data = "<data>")]
pub fn tell_me_who(
    data: Json<MirrorRequest>,
    sender: State<Arc<Sender<ImageRequestVectorized>>>,
    receiver: State<Arc<Receiver<Vec<Vec<Bbox>>>>>,
) -> RouterResponse {
    info!(
        "Received data json body with image base64 encoded image of length {:?}",
        &data.img.len()
    );
    if let Ok(bytes) = decode(&data.img) {
        info!("Received {:?} bytes", &bytes.len());
        let img_vectorized = ImageRequestVectorized {
            img: bytes,
            request: RequestType::BBOXES,
        };
        if let Ok(s) = sender.send(img_vectorized) {
            if let Ok(solution) = receiver.recv() {
                return RouterResponse::OkString(Json(solution));
            }
        }
    };
    RouterResponse::Error(format!("Error: Cannot handle operation."))
}

#[post("/showmewho", format = "image/jpeg", data = "<data>")]
pub fn show_me_who(
    data: Json<MirrorRequest>,
    sender: State<Arc<Sender<ImageRequestVectorized>>>,
    receiver: State<Receiver<Arc<Vec<u8>>>>,
) -> RouterResponse {
    // if let Ok(s) = sender.send(ImageRequestVectorized {
    //     img: image_buf,
    //     request: RequestType::IMAGE,
    // }) {
    //     println!("Data has been sent:{:?}", &s);
    // }
    RouterResponse::OkBinary(Vec::new())
}
