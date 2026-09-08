set shell := ["bash", "-c"]

lint:
    cargo fmt --check
    cargo run --release --quiet -p game --bin lint
    cargo clippy --release --all-targets -- -D warnings
    cargo clippy --release -p game --lib --bin client --target wasm32-unknown-unknown -- -D warnings

test:
    cargo test --release -p game

# Fixed fluid workload; prints microseconds per particle-substep.
bench:
    cargo run --release -p game --bin bench

# The browser client: a wasm binary post-processed by wasm-bindgen into the bundle the page loads.
# The `wasm` profile is the small, slow-to-build deploy; `release` is the fast local loop.
wasm profile="wasm":
    cargo build --profile {{profile}} -p game --bin client --target wasm32-unknown-unknown
    wasm-bindgen --target web --no-typescript --out-name spin --out-dir target/wasm \
      target/wasm32-unknown-unknown/{{profile}}/client.wasm

# Everything the static host serves: the page plus the wasm bundle.
dist profile="wasm": (wasm profile)
    rm -rf dist && mkdir -p dist
    cp static/index.html dist/
    cp target/wasm/spin.js target/wasm/spin_bg.wasm dist/

# Serve dist/ locally.
serve: dist
    python3 -m http.server --directory dist 8000

# Local development: a plain release build served on :8000. No hot reload — rerun and refresh.
dev: (dist "release")
    python3 -m http.server --directory dist 8000

# Load dist/ in headless Chrome and drive it through the page's script hooks.
e2e: dist
    node e2e/smoke.mjs
