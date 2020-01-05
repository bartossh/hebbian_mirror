FROM rust:latest
RUN rustup toolchain install nightly
RUN rustup default nightly
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app
RUN chmod +x /usr/src/app/assets/download.sh
WORKDIR /usr/src/app/assets
RUN ./download.sh
WORKDIR /usr/src/app
EXPOSE 8000
