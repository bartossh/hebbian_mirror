# Hebbian Mirror:

Low level http service for image recognition. Written in Rust with CUDA support.

<p align="center">
    <img 
    width="50%" height="50%" 
    src="https://github.com/Bartoshko/hebbian_mirror/blob/master/assets/mirror.jpeg"/>
</p>

## Dependencies:

This software requires:

- Rust nightly: rustc 1.40.0-nightly, cargo 1.40.0-nightly, rustup 1.20.2
- [Rocket](https://rocket.rs/)
- [tch](https://docs.rs/tch/0.1.1/tch/)
- [openssl](https://github.com/openssl/openssl)
- [Docker](https://www.docker.com/)
- and more (please see Cargo.toml)

- please if building for IoT use proper compilation settings or build on device.
- this software uses layer of virtualization only for development and QA process, which will effect running performance. Use docker 3.0 or greater,
- For AWS lambda integration I am going to create .env file instruction that will allow to build project on the go and be used with your AWS tools (but this is going to happened in near future)
- please compile with `IS_CUDA = false` first, check if your hardware is capable of CUDA

## Deep Learning assets:

- Download weights ```$ wget -c https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/yolo-v3.ot ```

## Build for development:

- cargo: ```$ RUST_LOG=info ROCKET_ENV=development cargo run```
- docker: ```docker-compose up```

## Build for production

- ```$ RUST_LOG=error ROCKET_ENV=production cargo run```

or 

- - ```$ cargo build --release``` 
then
```chmod +x ./target/release/hebbian_mirror```
then
```$ RUST_LOG=error ROCKET_ENV=production ./target/release/hebbian_mirror```

## Training and testing neuron network guide

We are using [Pytorch library](https://pytorch.org) for neuron network training
testing and validation.
To set up your environment for Python and Pytorch please install required packages:
- Install python3-tk, f.e: on Ubuntu based Linux: ```sudo apt-get install python3-tk```
- Set up virtual environment, f.e: on Ubuntu based Linux ```python3 -m venv venv```
- Set virtual environment: ```source venv/bin/activate```
- Install all python libraries: ```pip install ir requirements.txt```
- To get information how to run training / evaluating software run in terminal
```python main.py -h``` do help message will guide You.
- Test example: ```python main.py -p deeplabv3_resnet101 assets/fishing.jpeg```

## Authors and Contributors:

- Lenart Bartosz

## License:

- [MIT](https://opensource.org/licenses/MIT)
