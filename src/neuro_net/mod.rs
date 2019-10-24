use crate::failure;
use crate::tch::{
    nn,
    nn::{FuncT, ModuleT},
    Tensor,
};
use crate::{BTreeMap, BufRead, BufReader, File, Path};
use crate::{CONFIDENCE_THRESHOLD, NAMES, NMS_THRESHOLD};

#[derive(Debug, Clone)]
struct Block {
    block_type: String,
    parameters: BTreeMap<String, String>,
}

impl Block {
    fn get(&self, key: &str) -> failure::Fallible<&str> {
        match self.parameters.get(&key.to_string()) {
            None => bail!("cannot find {} in {}", key, self.block_type),
            Some(value) => Ok(value),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Darknet {
    blocks: Vec<Block>,
    parameters: BTreeMap<String, String>,
}

impl Darknet {
    fn get(&self, key: &str) -> failure::Fallible<&str> {
        match self.parameters.get(&key.to_string()) {
            None => bail!("cannot find {} in net parameters", key),
            Some(value) => Ok(value),
        }
    }
}

struct Accumulator {
    block_type: Option<String>,
    parameters: BTreeMap<String, String>,
    net: Darknet,
}

impl Accumulator {
    fn new() -> Accumulator {
        Accumulator {
            block_type: None,
            parameters: BTreeMap::new(),
            net: Darknet {
                blocks: vec![],
                parameters: BTreeMap::new(),
            },
        }
    }

    fn finish_block(&mut self) {
        match &self.block_type {
            None => (),
            Some(block_type) => {
                if block_type == "net" {
                    self.net.parameters = self.parameters.clone();
                } else {
                    let block = Block {
                        block_type: block_type.to_string(),
                        parameters: self.parameters.clone(),
                    };
                    self.net.blocks.push(block);
                }
                self.parameters.clear();
            }
        }
        self.block_type = None;
    }
}

pub fn parse_config<T: AsRef<Path>>(path: T) -> failure::Fallible<Darknet> {
    let file = File::open(path.as_ref())?;
    let mut acc = Accumulator::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.is_empty() || line.starts_with("#") {
            continue;
        }
        let line = line.trim();
        if line.starts_with("[") {
            ensure!(line.ends_with("]"), "line does not end with ']' {}", line);
            let line = &line[1..line.len() - 1];
            acc.finish_block();
            acc.block_type = Some(line.to_string());
        } else {
            let key_value: Vec<&str> = line.splitn(2, "=").collect();
            ensure!(key_value.len() == 2, "missing equal {}", line);
            let prev = acc.parameters.insert(
                key_value[0].trim().to_owned(),
                key_value[1].trim().to_owned(),
            );
            ensure!(prev == None, "multiple value for key {}", line);
        }
    }
    acc.finish_block();
    Ok(acc.net)
}

enum Bl {
    Layer(Box<dyn ModuleT>),
    Route(Vec<usize>),
    Shortcut(usize),
    Yolo(i64, Vec<(i64, i64)>),
}

fn conv(vs: nn::Path, index: usize, p: i64, b: &Block) -> failure::Fallible<(i64, Bl)> {
    let activation = b.get("activation")?;
    let filters = b.get("filters")?.parse::<i64>()?;
    let pad = b.get("pad")?.parse::<i64>()?;
    let size = b.get("size")?.parse::<i64>()?;
    let stride = b.get("stride")?.parse::<i64>()?;
    let pad = if pad != 0 { (size - 1) / 2 } else { 0 };
    let (bn, bias) = match b.parameters.get("batch_normalize") {
        Some(p) if p.parse::<i64>()? != 0 => {
            let vs = &vs / format!("batch_norm_{}", index);
            let bn = nn::batch_norm2d(&vs, filters, Default::default());
            (Some(bn), false)
        }
        Some(_) | None => (None, true),
    };
    let conv_cfg = nn::ConvConfig {
        stride,
        padding: pad,
        bias,
        ..Default::default()
    };
    let vs = &vs / format!("conv_{}", index);
    let conv = nn::conv2d(vs, p, filters, size, conv_cfg);
    let leaky = match activation {
        "leaky" => true,
        "linear" => false,
        otherwise => bail!("unsupported activation {}", otherwise),
    };
    let func = nn::func_t(move |xs, train| {
        let xs = xs.apply(&conv);
        let xs = match &bn {
            Some(bn) => xs.apply_t(bn, train),
            None => xs,
        };
        if leaky {
            xs.max1(&(&xs * 0.1))
        } else {
            xs
        }
    });
    Ok((filters, Bl::Layer(Box::new(func))))
}

fn upsample(prev_channels: i64) -> failure::Fallible<(i64, Bl)> {
    let layer = nn::func_t(|xs, _is_training| {
        let (_n, _c, h, w) = xs.size4().unwrap();
        xs.upsample_nearest2d(&[2 * h, 2 * w])
    });
    Ok((prev_channels, Bl::Layer(Box::new(layer))))
}

fn int_list_of_string(s: &str) -> failure::Fallible<Vec<i64>> {
    let res: Result<Vec<_>, _> = s.split(",").map(|xs| xs.trim().parse::<i64>()).collect();
    Ok(res?)
}

fn usize_of_index(index: usize, i: i64) -> usize {
    if i >= 0 {
        i as usize
    } else {
        (index as i64 + i) as usize
    }
}

fn route(index: usize, p: &Vec<(i64, Bl)>, block: &Block) -> failure::Fallible<(i64, Bl)> {
    let layers = int_list_of_string(block.get("layers")?)?;
    let layers: Vec<usize> = layers
        .into_iter()
        .map(|l| usize_of_index(index, l))
        .collect();
    let channels = layers.iter().map(|&l| p[l].0).sum();
    Ok((channels, Bl::Route(layers)))
}

fn shortcut(index: usize, p: i64, block: &Block) -> failure::Fallible<(i64, Bl)> {
    let from = block.get("from")?.parse::<i64>()?;
    Ok((p, Bl::Shortcut(usize_of_index(index, from))))
}

fn yolo(p: i64, block: &Block) -> failure::Fallible<(i64, Bl)> {
    let classes = block.get("classes")?.parse::<i64>()?;
    let flat = int_list_of_string(block.get("anchors")?)?;
    ensure!(flat.len() % 2 == 0, "even number of anchors");
    let anchors: Vec<_> = (0..(flat.len() / 2))
        .map(|i| (flat[2 * i], flat[2 * i + 1]))
        .collect();
    let mask = int_list_of_string(block.get("mask")?)?;
    let anchors = mask.into_iter().map(|i| anchors[i as usize]).collect();
    Ok((p, Bl::Yolo(classes, anchors)))
}

// Apply f to a slice of tensor xs and replace xs values with f output.
fn slice_apply_and_set<F>(xs: &mut Tensor, start: i64, len: i64, f: F)
where
    F: FnOnce(&Tensor) -> Tensor,
{
    let mut slice = xs.narrow(2, start, len);
    let src = f(&slice);
    slice.copy_(&src)
}

fn detect(xs: &Tensor, image_height: i64, classes: i64, anchors: &Vec<(i64, i64)>) -> Tensor {
    let (bsize, _channels, height, _width) = xs.size4().unwrap();
    let stride = image_height / height;
    let grid_size = image_height / stride;
    let bbox_attrs = 5 + classes;
    let nanchors = anchors.len() as i64;
    let mut xs = xs
        .view((bsize, bbox_attrs * nanchors, grid_size * grid_size))
        .transpose(1, 2)
        .contiguous()
        .view((bsize, grid_size * grid_size * nanchors, bbox_attrs));
    let grid = Tensor::arange(grid_size, tch::kind::FLOAT_CPU);
    let a = grid.repeat(&[grid_size, 1]);
    let b = a.tr().contiguous();
    let x_offset = a.view((-1, 1));
    let y_offset = b.view((-1, 1));
    let xy_offset = Tensor::cat(&[x_offset, y_offset], 1)
        .repeat(&[1, nanchors])
        .view((-1, 2))
        .unsqueeze(0);
    let anchors: Vec<f32> = anchors
        .iter()
        .flat_map(|&(x, y)| vec![x as f32 / stride as f32, y as f32 / stride as f32].into_iter())
        .collect();
    let anchors = Tensor::of_slice(&anchors)
        .view((-1, 2))
        .repeat(&[grid_size * grid_size, 1])
        .unsqueeze(0);
    slice_apply_and_set(&mut xs, 0, 2, |xs| xs.sigmoid() + xy_offset);
    slice_apply_and_set(&mut xs, 4, 1 + classes, Tensor::sigmoid);
    slice_apply_and_set(&mut xs, 2, 2, |xs| xs.exp() * anchors);
    slice_apply_and_set(&mut xs, 0, 4, |xs| xs * stride);
    xs
}

impl Darknet {
    pub fn height(&self) -> failure::Fallible<i64> {
        let image_height = self.get("height")?.parse::<i64>()?;
        Ok(image_height)
    }

    pub fn width(&self) -> failure::Fallible<i64> {
        let image_width = self.get("width")?.parse::<i64>()?;
        Ok(image_width)
    }

    pub fn build_model(&self, vs: &nn::Path) -> failure::Fallible<FuncT> {
        let mut blocks: Vec<(i64, Bl)> = vec![];
        let mut prev_channels: i64 = 3;
        for (index, block) in self.blocks.iter().enumerate() {
            let channels_and_bl = match block.block_type.as_str() {
                "convolutional" => conv(vs / index, index, prev_channels, &block)?,
                "upsample" => upsample(prev_channels)?,
                "shortcut" => shortcut(index, prev_channels, &block)?,
                "route" => route(index, &blocks, &block)?,
                "yolo" => yolo(prev_channels, &block)?,
                otherwise => bail!("unsupported block type {}", otherwise),
            };
            prev_channels = channels_and_bl.0;
            blocks.push(channels_and_bl);
        }
        let image_height = self.height()?;
        let func = nn::func_t(move |xs, train| {
            let mut prev_ys: Vec<Tensor> = vec![];
            let mut detections: Vec<Tensor> = vec![];
            for (_, b) in blocks.iter() {
                let ys = match b {
                    Bl::Layer(l) => {
                        let xs = prev_ys.last().unwrap_or(&xs);
                        l.forward_t(&xs, train)
                    }
                    Bl::Route(layers) => {
                        let layers: Vec<_> = layers.iter().map(|&i| &prev_ys[i]).collect();
                        Tensor::cat(&layers, 1)
                    }
                    Bl::Shortcut(from) => prev_ys.last().unwrap() + prev_ys.get(*from).unwrap(),
                    Bl::Yolo(classes, anchors) => {
                        let xs = prev_ys.last().unwrap_or(&xs);
                        detections.push(detect(xs, image_height, *classes, anchors));
                        Tensor::default()
                    }
                };
                prev_ys.push(ys);
            }
            Tensor::cat(&detections, 1)
        });
        Ok(func)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Bbox {
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
    confidence: f64,
    class_confidence: f64,
}

// Intersection over union of two bounding boxes.
fn iou(b1: &Bbox, b2: &Bbox) -> f64 {
    let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
    let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
    let i_xmin = b1.xmin.max(b2.xmin);
    let i_xmax = b1.xmax.min(b2.xmax);
    let i_ymin = b1.ymin.max(b2.ymin);
    let i_ymax = b1.ymax.min(b2.ymax);
    let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
    i_area / (b1_area + b2_area - i_area)
}

pub fn report(pred: &Tensor, img: &Tensor, w: i64, h: i64) -> failure::Fallible<Vec<Vec<Bbox>>> {
    let (npreds, pred_size) = pred.size2()?;
    let nclasses = (pred_size - 5) as usize;
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred = Vec::<f64>::from(pred.get(index));
        let confidence = pred[4];
        if confidence > CONFIDENCE_THRESHOLD {
            let mut class_index = 0;
            for i in 0..nclasses {
                if pred[5 + i] > pred[5 + class_index] {
                    class_index = i
                }
            }
            if pred[class_index + 5] > 0. {
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                    class_confidence: pred[5 + class_index],
                };
                bboxes[class_index].push(bbox)
            }
        }
    }
    // Perform non-maximum suppression.
    for bboxes_for_class in bboxes.iter_mut() {
        bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
        let mut current_index = 0;
        for index in 0..bboxes_for_class.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);
                if iou > NMS_THRESHOLD {
                    drop = true;
                    break;
                }
            }
            if !drop {
                bboxes_for_class.swap(current_index, index);
                current_index += 1;
            }
        }
        bboxes_for_class.truncate(current_index);
    }
    let (_, initial_h, initial_w) = img.size3()?;
    let w_ratio = initial_w as f64 / w as f64;
    let h_ratio = initial_h as f64 / h as f64;
    let mut bboxes_scaled: Vec<Vec<Bbox>> = Vec::new();
    for (_, bboxes_for_class) in bboxes.iter().enumerate() {
        let mut bboxes_for_class_scaled: Vec<Bbox> = Vec::new();
        for b in bboxes_for_class.iter() {
            let xmin = ((b.xmin * w_ratio) as i64).max(0).min(initial_w - 1);
            let ymin = ((b.ymin * h_ratio) as i64).max(0).min(initial_h - 1);
            let xmax = ((b.xmax * w_ratio) as i64).max(0).min(initial_w - 1);
            let ymax = ((b.ymax * h_ratio) as i64).max(0).min(initial_h - 1);
            bboxes_for_class_scaled.push(Bbox {
                xmin: xmin as f64,
                ymin: ymin as f64,
                xmax: xmax as f64,
                ymax: ymax as f64,
                confidence: b.confidence,
                class_confidence: b.class_confidence,
            });
        }
        bboxes_scaled.push(bboxes_for_class_scaled);
    }
    Ok(bboxes_scaled)
}

pub fn draw_results(img: &Tensor, bboxes: Vec<Vec<Bbox>>) -> failure::Fallible<Tensor> {
    let (_, initial_h, initial_w) = img.size3()?;
    let mut img = img.to_kind(tch::Kind::Float) / 255.;
    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
        for b in bboxes_for_class.iter() {
            println!("{}: {:?}", NAMES[class_index], b);
            draw_rect(
                &mut img,
                b.xmin as i64,
                b.xmax as i64,
                b.ymin as i64,
                b.ymax.min(b.ymin + 2.) as i64,
            );
            draw_rect(
                &mut img,
                b.xmin as i64,
                b.xmax as i64,
                b.ymin.max(b.ymax - 2.) as i64,
                b.ymax as i64,
            );
            draw_rect(
                &mut img,
                b.xmin as i64,
                b.xmax.min(b.xmin + 2.) as i64,
                b.ymin as i64,
                b.ymax as i64,
            );
            draw_rect(
                &mut img,
                b.xmin.max(b.xmax - 2.) as i64,
                b.xmax as i64,
                b.ymin as i64,
                b.ymax as i64,
            );
        }
    }
    Ok((img * 255.).to_kind(tch::Kind::Uint8))
}

// Assumes x1 <= x2 and y1 <= y2
pub fn draw_rect(t: &mut Tensor, x1: i64, x2: i64, y1: i64, y2: i64) {
    let color = Tensor::of_slice(&[0., 0., 1.]).view([3, 1, 1]);
    t.narrow(2, x1, x2 - x1)
        .narrow(1, y1, y2 - y1)
        .copy_(&color)
}
