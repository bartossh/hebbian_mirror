use crate::rocket::State;
use crate::{Arc, Bbox, Receiver, Sender};

#[derive(Debug, Clone)]
pub enum RequestType {
    BBOXES,
    IMAGE,
}

#[derive(Debug, Clone)]
pub struct ImageRequest {
    pub img: Vec<u8>,
    pub request: RequestType,
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

#[post("/tellmewho", format = "image/jpeg", data = "<image_buf>")]
pub fn tell_me_who(
    image_buf: Vec<u8>,
    sender: State<Arc<Sender<ImageRequest>>>,
    receiver: State<Arc<Receiver<Vec<Vec<Bbox>>>>>,
) -> RouterResponse {
    if let Ok(s) = sender.send(ImageRequest {
        img: image_buf,
        request: RequestType::BBOXES,
    }) {
        if let Ok(solution) = receiver.recv() {
            println!("solution : {:?}", &solution);
        }
    }
    RouterResponse::OkString(String::from(
        "You are the most beautiful on the whole world...",
    ))
}

#[post("/showmewho", format = "image/jpeg", data = "<image_buf>")]
pub fn show_me_who(
    image_buf: Vec<u8>,
    sender: State<Arc<Sender<ImageRequest>>>,
    receiver: State<Receiver<Arc<Vec<u8>>>>,
) -> RouterResponse {
    if let Ok(s) = sender.send(ImageRequest {
        img: image_buf,
        request: RequestType::IMAGE,
    }) {
        println!("Data has been sent:{:?}", &s);
    }
    RouterResponse::OkString(String::from(
        "There a Vec<u8> stream of the picture with bboxes around recognized objects will be returned, of curse You are beautiful",
    ))
}
