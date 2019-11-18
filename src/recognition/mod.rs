use crate::{
    delete_file, draw_results, image_helper, neuro_net, report, save_file, stop_program,
    tch::nn::FuncT, Bbox, ImageRequestVectorized, ModuleT, Receiver, RequestType, Sender, Tensor,
    VarStore,
};
use crate::{CONFIG, CUDA_THREADS, IS_CUDA, TEMPORARY_FILE_PATH, WEIGHTS};

/// Runs single thread listener that takes image recognition request from unbounded pipe
/// and based on request type performs recognition, then sends solution to one of two unbounded pipes.
/// Which one of two pipes is chosen depends on request type.
///
/// If RequestType matches BBOXES than sender_plane_bboxed pipe receives solution in Bbox format
/// If RequestType matches IMAGE than sender_image_bboxed pipe receives solution in Vec<u8> format
///
/// # Argument
///
/// * receiver_img - receiver pipe, to receive requests through
/// * sender_plane_bboxed - pipe to send vectorized Bbox solution through
/// * sender_imgage_bboxed - pipe to send binary buffer solution through
///
pub fn run_recognition_ai_listener(
    receiver_img: Receiver<ImageRequestVectorized>,
    sender_plane_bboxed: Sender<Vec<Vec<Bbox>>>,
    sender_imgage_bboxed: Sender<Vec<u8>>,
) {
    let mut var_store: VarStore = VarStore::new(tch::Device::Cpu);
    if IS_CUDA {
        var_store = VarStore::new(tch::Device::Cuda(CUDA_THREADS));
    }
    match neuro_net::parse_config(&CONFIG.to_string()) {
        Ok(darknet) => {
            let net_width = darknet.width().expect("Cannot get darknet width");
            let net_height = darknet.height().expect("Cannot get darknet height");
            match darknet.build_model(&var_store.root()) {
                Ok(model) => {
                    if let Err(_) = var_store.load(&WEIGHTS.to_string()) {
                        stop_program();
                    };
                    looper_listener(
                        receiver_img,
                        sender_plane_bboxed,
                        sender_imgage_bboxed,
                        model,
                        net_width,
                        net_height,
                    );
                }
                Err(_) => stop_program(),
            };
        }
        Err(_) => {
            stop_program();
        }
    };
}

fn looper_listener(
    receiver_img: Receiver<ImageRequestVectorized>,
    sender_plane_bboxed: Sender<Vec<Vec<Bbox>>>,
    sender_imgage_bboxed: Sender<Vec<u8>>,
    model: FuncT,
    net_width: i64,
    net_height: i64,
) {
    loop {
        if let Ok(img_request) = receiver_img.recv() {
            if let Err(_) = save_file(&img_request.img, &TEMPORARY_FILE_PATH.to_string()) {
                panic!("It is not possible to save temporary file, something went terribly wrong.");
            } else {
                let original_image = image_helper::load(&TEMPORARY_FILE_PATH.to_string())
                    .expect("Cannot load image from file");
                let image_resized: Tensor =
                    image_helper::resize(&original_image, net_width, net_height)
                        .expect("Cannot resize image");
                let image_calc_ready: Tensor =
                    image_resized.unsqueeze(0).to_kind(tch::Kind::Float) / 255.;
                let predictions = model.forward_t(&image_calc_ready, false).squeeze();
                let bboxes = report(&predictions, &original_image, net_width, net_height)
                    .expect("Cannot generate report");
                match img_request.request {
                    RequestType::BBOXES => {
                        if let Err(_) = sender_plane_bboxed.send(bboxes) {
                            error!("Cannot send bboxes solution between threads.");
                        }
                    }
                    RequestType::IMAGE => {
                        if let Ok(img_drawed_boxes) = draw_results(&image_resized, bboxes) {
                            let image_buffer: Vec<u8> = Vec::from(img_drawed_boxes);
                            info!(
                                "Copy tensor to image buffer of length {:?}",
                                &image_buffer.len()
                            );
                            if let Err(_) = sender_imgage_bboxed.send(image_buffer) {
                                error!("Cannot send image buffer solution between threads.");
                            }
                        };
                    }
                };
                if let Err(_) = delete_file(&TEMPORARY_FILE_PATH.to_string()) {
                    panic!("It is not possible to delete saved temporary file, something went terribly wrong.");
                }
            }
        }
    }
}
