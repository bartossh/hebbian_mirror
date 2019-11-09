use crate::{
    delete_file, draw_results, image_helper, neuro_net, report, save_file, stop_program, Bbox,
    ImageRequestVectorized, ModuleT, Receiver, RequestType, Sender, Tensor, VarStore,
};
use crate::{
    CONFIDENCE_THRESHOLD, CONFIG, CUDA_THREADS, IS_CUDA, NAMES, NMS_THRESHOLD, TEMPORARY_FILE_PATH,
    WEIGHTS,
};

pub fn run_recognition_ai(
    receiver_img: Receiver<ImageRequestVectorized>,
    sender_plane_bboxed: Sender<Vec<Vec<Bbox>>>,
    sender_imgage_bboxed: Sender<Vec<u8>>,
) {
    let mut var_store: VarStore = VarStore::new(tch::Device::Cpu);
    if IS_CUDA {
        var_store = VarStore::new(tch::Device::Cuda(CUDA_THREADS));
    }
    if let Ok(darknet) = neuro_net::parse_config(&CONFIG.to_string()) {
        let net_width = darknet.width().expect("Cannot get darknet width");
        let net_height = darknet.height().expect("Cannot get darknet height");
        if let Ok(model) = darknet.build_model(&var_store.root()) {
            if let Err(_) = var_store.load(&WEIGHTS.to_string()) {
                stop_program();
            };
            loop {
                if let Ok(img_request) = receiver_img.recv() {
                    if let Ok(_) = save_file(&img_request.img, &TEMPORARY_FILE_PATH.to_string()) {
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
                                        error!(
                                            "Cannot send image buffer solution between threads."
                                        );
                                    }
                                };
                            }
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
}
