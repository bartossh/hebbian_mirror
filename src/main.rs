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

use crate::tch::nn::{FuncT, VarStore};
use crossbeam_channel::{unbounded, Receiver, Sender};
use settings::{CONFIDENCE_THRESHOLD, CONFIG, NAMES, NMS_THRESHOLD, WEIGHTS};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use std::thread;

fn main() {
    env_logger::init();
    let (sender, receiver): (Sender<Vec<u8>>, Receiver<Vec<u8>>) = unbounded();
    let mut var_store = VarStore::new(tch::Device::Cpu);
    if let Ok(darknet) = neuro_net::parse_config(&CONFIG.to_string()) {
        let _net_width = darknet.width().expect("Cannot get darknet width");
        let _net_height = darknet.height().expect("Cannot get darknet height");
        if let Ok(_model) = darknet.build_model(&var_store.root()) {
            if let Err(_) = var_store.load(&WEIGHTS.to_string()) {
                stop_program();
            }
            println!("model {:?}", &_model);
        } else {
            stop_program();
        }
    } else {
        stop_program();
    }
    thread::spawn(move || loop {
        if let Ok(img) = receiver.recv() {
            println!("receiving {:?} ...", &img);
        }
    });
    println!("Starting server ....");
    rocket::ignite()
        .manage(Arc::new(sender.clone()))
        .mount("/mirror", routes![router::tell_me_who, router::show_me_who])
        .launch();
}

fn stop_program() {
    panic!(format!(
        "Fatal error, darknet cannot be loaded, please look in to documentation to check if all steps has been taken."
    ));
}
