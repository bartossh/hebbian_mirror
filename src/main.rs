#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use]
extern crate rocket;
#[macro_use]
extern crate log;
extern crate env_logger;

mod router;

fn main() {
    env_logger::init();
    info!("Starting service...");
    rocket::ignite()
        .mount("/mirror", routes![router::tell_me_who])
        .launch();
}
