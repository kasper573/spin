set shell := ["bash", "-c"]

lint:
    cargo fmt --check
    cargo run --release --quiet -p game --bin lint
    cargo clippy --release --all-targets -- -D warnings
    cargo clippy --release -p game --lib --bin client --target wasm32-unknown-unknown -- -D warnings

# What a machine without a GPU can afford.
test:
    cargo test --release -p game

# The game played as a player plays it, every frame drawn, on this machine's GPU. Each scene
# leaves its frames, a sheet of them and what it measured in target/playgate/<scene>/. A scene
# that passes has only kept its numbers: whether the game looks and plays right is seen on the
# sheets, which are there to be looked at.
gate:
    cargo test --release -p game --test play_gate --no-fail-fast -- --ignored --test-threads=1; status=$?; \
    for scene in target/playgate/*/; do \
      ffmpeg -y -loglevel error -pattern_type glob -i "${scene}frame_*.png" \
        -vf "select='not(mod(n\,7))',scale=640:360,tile=4x3" -frames:v 1 "${scene}sheet.png"; \
    done; exit $status

# The whole gate before pushing: what the runners check, plus everything they cannot afford.
verify: lint test gate e2e

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

# The same client in a native window, at native speed, for playing and profiling.
dev-native:
    cargo run --release -p game --bin client

# Load dist/ in headless Chrome and drive it through the page's script hooks.
e2e: dist
    node e2e/smoke.mjs
