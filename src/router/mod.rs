use crate::rocket::State;
use crate::{Arc, Sender};

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
pub fn tell_me_who(sender: State<Arc<Sender<Vec<u8>>>>) -> RouterResponse {
    if let Ok(s) = sender.send(vec![0, 1, 2, 3, 4, 5, 6]) {
        println!("{:?}", &s);
    }
    RouterResponse::OkString(String::from(
        "You are the most beautiful on the whole world...",
    ))
}

#[post("/showmewho")]
pub fn show_me_who(sender: State<Arc<Sender<Vec<u8>>>>) -> RouterResponse {
    RouterResponse::OkString(String::from(
        "There a Vec<u8> stream of the picture with bboxes around recognized objects will be returned",
    ))
}
