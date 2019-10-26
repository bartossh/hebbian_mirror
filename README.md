# Hebbian Mirror:

Low level http service for image recognition. Written in Rust with CUDA support.

<p align="center"><img width="50%" height="50%" src="https://github.com/Bartoshko/hebbian_mirror/blob/master/assets/mirror.jpeg" height="100%" width="100%"/></p>

## Dependencies:

This software requires:

- Rust nightly: rustc 1.40.0-nightly, cargo 1.40.0-nightly, rustup 1.20.2
- [Rocket](https://rocket.rs/)
- [tch](https://docs.rs/tch/0.1.1/tch/)
- [openssl](https://github.com/openssl/openssl)
- and more (please see Cargo.toml)

- please if building for IoT use proper compilation settings or build on device.
- this software does not use any layer of virtualization to offer best possible performance
- For AWS lambda integration I am going to create .env file instruction that will allow to build project on the go and be used with your AWS tools (but this is going to happened in near future)

## Deep Learning assets:

- Download weights ```$ wget -c https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/yolo-v3.ot ```

## Build for development:

- ```$ RUST_ENV=info ROCKET_ENV=development cargo run```

## Build for production

- ```$ RUST_ENV=error ROCKET_ENV=production cargo run```

or 

- - ```$ cargo build --release``` 
then
```chmod +x ./target/release/hebbian_mirror```
then
```$ RUST_ENV=error ROCKET_ENV=production ./target/release/hebbian_mirror```

## Authors and Contributors:

- Lenart Bartosz

## License:

- [MIT](https://opensource.org/licenses/MIT)
