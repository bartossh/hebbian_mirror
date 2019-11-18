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
extern crate crossbeam_channel;
extern crate tch;

mod helpers;
mod neuro_net;
mod recognition;
mod router;
mod settings;

use crossbeam_channel::{unbounded, Receiver, Sender};
use helpers::{delete_file, save_file, stop_program};
use neuro_net::{draw_results, report, Bbox};
use recognition::run_recognition_ai_listener;
use rocket_contrib::json::Json;
use router::{ImageRequestVectorized, RequestType};
use settings::{
    CONFIDENCE_THRESHOLD, CONFIG, CUDA_THREADS, IS_CUDA, NAMES, NMS_THRESHOLD, TEMPORARY_FILE_PATH,
    WEIGHTS,
};
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
use tch::Tensor;

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
    let (sender_imgage_bboxed, receiver_image_bboxed): (Sender<Vec<u8>>, Receiver<Vec<u8>>) =
        unbounded();
    thread::spawn(move || {
        run_recognition_ai_listener(
            receiver_img.clone(),
            sender_plane_bboxed.clone(),
            sender_imgage_bboxed.clone(),
        )
    });
    rocket::ignite()
        .manage(Arc::new(sender_img.clone()))
        .manage(Arc::new(receiver_plane_bboxed.clone()))
        .manage(Arc::new(receiver_image_bboxed.clone()))
        .mount(
            "/recognition/object",
            routes![
                router::get,
                router::get_names,
                router::post_recognize_objects_boxes,
                router::post_recognize_objects_images
            ],
        )
        .launch();
}
