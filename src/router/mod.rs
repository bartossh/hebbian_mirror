use crate::rocket::State;
use crate::{Arc, Bbox, Json, Receiver, Sender};

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

#[derive(Debug, Clone, Deserialize)]
pub struct Base64Image {
    image: String,
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

#[post("/tellmewho", format = "json", data = "<image_post>")]
pub fn tell_me_who(
    image_post: Json<Base64Image>,
    sender: State<Arc<Sender<ImageRequest>>>,
    receiver: State<Arc<Receiver<Vec<Vec<Bbox>>>>>,
) -> RouterResponse {
    println!("{:?}", image_post);
    // TODO: here get image from base64 to vector or write down in to file
    //let image_request = ImageRequest {img: image_buf, request: RequestType::BBOXES};
    //if let Ok(s) = sender.send(image_request) {
    //    if let Ok(solution) = receiver.recv() {
    //        println!("solution : {:?}", &solution);
    //    }
    //}
    RouterResponse::OkString(String::from(
        "Prosopagnosia is being a pro in so called agnosticism",
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
        "Prosopagnosia is being a pro in so called agnosticism",
    ))
}
