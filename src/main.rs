#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use]
extern crate rocket;
extern crate rocket_contrib;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate log;
extern crate env_logger;
#[macro_use]
extern crate failure;
extern crate base64;
extern crate crossbeam_channel;
extern crate tch;

mod helpers;
mod neuro_net;
mod router;
mod settings;

use base64::decode;
use crossbeam_channel::{unbounded, Receiver, Sender};
use helpers::{delete_file, save_file};
use neuro_net::{report, Bbox};
use rocket_contrib::json::Json;
use router::{ImageRequestVectorized, RequestType};
use settings::{CONFIDENCE_THRESHOLD, CONFIG, NAMES, NMS_THRESHOLD, TEMPORARY_FILE_PATH, WEIGHTS};
use std::collections::BTreeMap;
use std::fs::remove_file;
use std::fs::File;
use std::io::prelude::Write;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use std::thread;
use tch::nn::{ModuleT, VarStore};
use tch::vision::image as image_helper;

fn main() {
    env_logger::init();
    let (sender_img, receiver_img): (
        Sender<ImageRequestVectorized>,
        Receiver<ImageRequestVectorized>,
    ) = unbounded();
    let (sender_plane_bboxed, receiver_plane_bboxed): (
        Sender<Vec<Vec<Bbox>>>,
        Receiver<Vec<Vec<Bbox>>>,
    ) = unbounded();
    let (_sender_img_bboxed, receiver_image_bboxed): (
        Sender<Arc<Vec<u8>>>,
        Receiver<Arc<Vec<u8>>>,
    ) = unbounded();
    thread::spawn(move || {
        let mut var_store = VarStore::new(tch::Device::Cpu);
        if let Ok(darknet) = neuro_net::parse_config(&CONFIG.to_string()) {
            let net_width = darknet.width().expect("Cannot get darknet width");
            let net_height = darknet.height().expect("Cannot get darknet height");
            if let Ok(model) = darknet.build_model(&var_store.root()) {
                if let Err(_) = var_store.load(&WEIGHTS.to_string()) {
                    stop_program();
                };
                loop {
                    if let Ok(img_request) = receiver_img.recv() {
                        if let Ok(_) = save_file(&img_request.img, &TEMPORARY_FILE_PATH.to_string())
                        {
                            info!("Buffer temporally saved with success.");
                            info!(
                                "Moving image buffer of length {:?} between threads",
                                &img_request.img.len()
                            );

                            let original_image =
                                image_helper::load(&TEMPORARY_FILE_PATH.to_string())
                                    .expect("Cannot load image from file");
                            let image =
                                image_helper::resize(&original_image, net_width, net_height)
                                    .expect("Cannot resize image");
                            let image = image.unsqueeze(0).to_kind(tch::Kind::Float) / 255.;
                            let predictions = model.forward_t(&image, false).squeeze();
                            let bboxes =
                                report(&predictions, &original_image, net_width, net_height)
                                    .expect("Cannot generate report");
                            match img_request.request {
                                RequestType::BBOXES => {
                                    if let Err(_) = sender_plane_bboxed.send(bboxes) {
                                        error!("Cannot send solution between threads.");
                                    }
                                }
                                RequestType::IMAGE => {}
                            };
                            if let Err(_) = delete_file(&TEMPORARY_FILE_PATH.to_string()) {
                                panic!("It is not possible to delete saved temporary file, something went terribly wrong.");
                            }
                        } else {
                            panic!("It is not possible to save temporary file, something went terribly wrong.");
                        }
                    }
                }
            } else {
                stop_program();
            }
        } else {
            stop_program();
        }
    });
    println!("Starting server ....");
    rocket::ignite()
        .manage(Arc::new(sender_img.clone()))
        .manage(Arc::new(receiver_plane_bboxed.clone()))
        .manage(Arc::new(receiver_image_bboxed.clone()))
        .mount(
            "/mirror",
            routes![router::tell_me_who, router::show_me_who, router::get_names],
        )
        .launch();
}

fn stop_program() {
    panic!(format!(
        "Fatal error, darknet cannot be loaded, please look in to documentation to check if all steps has been taken."
    ));
}
