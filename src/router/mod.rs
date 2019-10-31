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
    OkString(String),
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
    println!("{:?}", &data.img);
    if let Ok(bytes) = decode(&data.img) {
        println!("Received {:?} bytes", &bytes.len());
        let img_vectorized = ImageRequestVectorized {
            img: bytes,
            request: RequestType::BBOXES,
        };
        if let Ok(s) = sender.send(img_vectorized) {
            if let Ok(solution) = receiver.recv() {
                println!("solution : {:?}", &solution);
            }
        }
    };
    RouterResponse::OkString(String::from(
        "Prosopagnosia is being a pro in so called agnosticism",
    ))
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
    RouterResponse::OkString(String::from(
        "Prosopagnosia is being a pro in so called agnosticism",
    ))
}
