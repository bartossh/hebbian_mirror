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
pub fn tell_me_who() -> RouterResponse {
    RouterResponse::OkString(String::from(
        "You are the most beautiful on the whole world...",
    ))
}
