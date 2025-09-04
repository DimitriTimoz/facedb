FROM rustlang/rust:nightly as builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
# Create a dummy src to cache dependencies
RUN mkdir src && echo "fn main(){}" > src/main.rs
RUN cargo build --release || true
COPY . ./
RUN cargo build --release

FROM debian:bookworm-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/facedb /app/facedb
RUN mkdir -p /app/images
EXPOSE 8080
ENV RUST_LOG=info
CMD ["/app/facedb"]
