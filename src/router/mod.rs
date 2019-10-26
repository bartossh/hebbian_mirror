use crate::rocket::State;
use crate::{Arc, ImageRequest, RequestType, Sender};

#[derive(Responder)]
pub enum RouterResponse {
    #[response(status = 200, content_type = "application/json")]
    OkString(String),
    #[response(status = 400)]
    NotFound(String),
    #[response(status = 500, content_type = "plain")]
    Error(String),
}

#[post("/tellmewho")]
pub fn tell_me_who(sender: State<Arc<Sender<ImageRequest>>>) -> RouterResponse {
    if let Ok(s) = sender.send(ImageRequest {
        img: vec![0, 1, 2, 3, 4, 5, 6],
        request: RequestType::RAW,
    }) {
        println!("{:?}", &s);
    }
    RouterResponse::OkString(String::from(
        "You are the most beautiful on the whole world...",
    ))
}

#[post("/showmewho")]
pub fn show_me_who(sender: State<Arc<Sender<ImageRequest>>>) -> RouterResponse {
    RouterResponse::OkString(String::from(
        "There a Vec<u8> stream of the picture with bboxes around recognized objects will be returned, of curse You are beautiful",
    ))
}
