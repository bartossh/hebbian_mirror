#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use]
extern crate rocket;
#[macro_use]
extern crate log;
extern crate env_logger;
#[macro_use]
extern crate failure;
extern crate crossbeam_channel;
extern crate tch;

mod neuro_net;
mod router;
mod settings;

use crossbeam_channel::{unbounded, Receiver, Sender};
use neuro_net::{report, Bbox};
use router::{ImageRequest, RequestType};
use settings::{CONFIDENCE_THRESHOLD, CONFIG, NAMES, NMS_THRESHOLD, WEIGHTS};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use std::thread;
use tch::nn::{ModuleT, VarStore};
use tch::vision::image as image_helper;
use tch::Tensor;

fn main() {
    env_logger::init();
    let (sender_img, receiver_img): (Sender<ImageRequest>, Receiver<ImageRequest>) = unbounded();
    let (sender_plane_bboxed, receiver_plane_bboxed): (
        Sender<Vec<Vec<Bbox>>>,
        Receiver<Vec<Vec<Bbox>>>,
    ) = unbounded();
    let (sender_img_bboxed, receiver_image_bboxed): (Sender<Arc<Vec<u8>>>, Receiver<Arc<Vec<u8>>>) =
        unbounded();
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
                        match img_request.request {
                            RequestType::BBOXES => {
                                println!(
                                    "receiving for calculating BBOXES {:?} ... {:?}",
                                    &model, &img_request
                                );
                                let tensor_image = Tensor::of_slice(&img_request.img);
                                println!("DEBUG {:?}", tensor_image);
                                let image =
                                    image_helper::resize(&tensor_image, net_width, net_height)
                                        .expect("Cannot resize image");
                                println!("DEBUG {:?}", image);
                                let image = image.unsqueeze(0).to_kind(tch::Kind::Float) / 255.;
                                let predictions = model.forward_t(&image, false).squeeze();
                                let bboxes =
                                    report(&predictions, &tensor_image, net_width, net_height)
                                        .expect("Cannot generate report");
                                info!("\n{:?}\n", bboxes);
                                if let Ok(_) = sender_plane_bboxed.send(bboxes) {
                                    println!("bboxes has been sent between threads");
                                }
                            }
                            RequestType::IMAGE => {
                                println!(
                                    "receiving for calculating IMAGES {:?} ... {:?}",
                                    &model, &img_request
                                );
                            }
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
        .mount("/mirror", routes![router::tell_me_who, router::show_me_who])
        .launch();
}

fn stop_program() {
    panic!(format!(
        "Fatal error, darknet cannot be loaded, please look in to documentation to check if all steps has been taken."
    ));
}
