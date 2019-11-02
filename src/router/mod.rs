use crate::rocket::State;
use crate::{decode, encode, Arc, Bbox, Json, Receiver, Sender, NAMES};

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
pub struct MirrorBase64Request {
    img: String,
}
#[derive(Debug, Clone, Serialize)]
pub struct MirrorBase64Response {
    img: String,
}

#[derive(Responder)]
pub enum RouterResponse {
    #[response(status = 200, content_type = "application/json")]
    OkBboxes(Json<Vec<Vec<Bbox>>>),
    #[response(status = 200, content_type = "application/json")]
    OkNames(Json<Vec<String>>),
    #[response(status = 200, content_type = "application/json")]
    OkBase64(Json<MirrorBase64Response>),
    #[response(status = 400)]
    NotFound(String),
    #[response(status = 500, content_type = "plain")]
    Error(String),
}

#[get("/names", format = "json")]
pub fn get_names() -> RouterResponse {
    let mut names = Vec::new();
    NAMES.iter().for_each(|name| {
        names.push(format!("{}", name));
    });
    RouterResponse::OkNames(Json(names))
}

#[post("/tellmewho", format = "json", data = "<data>")]
pub fn tell_me_who(
    data: Json<MirrorBase64Request>,
    sender: State<Arc<Sender<ImageRequestVectorized>>>,
    receiver: State<Arc<Receiver<Vec<Vec<Bbox>>>>>,
) -> RouterResponse {
    info!(
        "Received http request with data json body, with image base64 encoded image of length {:?}",
        &data.img.len()
    );
    if let Ok(bytes) = decode(&data.img) {
        info!("Received {:?} bytes", &bytes.len());
        let img_vectorized = ImageRequestVectorized {
            img: bytes,
            request: RequestType::BBOXES,
        };
        if let Ok(_) = sender.send(img_vectorized) {
            if let Ok(solution) = receiver.recv() {
                return RouterResponse::OkBboxes(Json(solution));
            }
        }
    };
    RouterResponse::Error(format!("Error: Cannot handle operation."))
}

#[post("/showmewho", format = "json", data = "<data>")]
pub fn show_me_who(
    data: Json<MirrorBase64Request>,
    sender: State<Arc<Sender<ImageRequestVectorized>>>,
    receiver: State<Arc<Receiver<Vec<u8>>>>,
) -> RouterResponse {
    info!(
        "Received http request with data json body, with image base64 encoded image of length {:?}",
        &data.img.len()
    );
    if let Ok(bytes) = decode(&data.img) {
        info!("Received {:?} bytes", &bytes.len());
        let img_vectorized = ImageRequestVectorized {
            img: bytes,
            request: RequestType::IMAGE,
        };
        if let Ok(_) = sender.send(img_vectorized) {
            if let Ok(solution_img_buffer) = receiver.recv() {
                let base64_string = encode(&solution_img_buffer);
                let response = MirrorBase64Response { img: base64_string };
                return RouterResponse::OkBase64(Json(response));
            }
        }
    };
    RouterResponse::Error(format!("Error: Cannot handle operation."))
}
