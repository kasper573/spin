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
#
# The driver holds the GPU at a fraction of its speed while the display sleeps, so the display
# is kept awake for as long as the scenes play, and the GPU's clocks are kept beside what was
# measured: a run in which the GPU worked below its full performance state measured the driver's
# thrift rather than the game, and fails.
gate: idle
    #!/usr/bin/env bash
    (while true; do xset dpms force on; sleep 20; done) & awake=$!
    nvidia-smi --query-gpu=pstate,utilization.gpu,clocks.gr --format=csv,noheader,nounits -l 2 \
      > target/playgate-gpu.csv & clocks=$!
    sleep 3
    cargo test --release -p game --test play_gate --no-fail-fast -- --ignored --test-threads=1
    status=$?
    kill $awake $clocks
    for scene in target/playgate/*/; do
      ffmpeg -y -loglevel error -pattern_type glob -i "${scene}frame_*.png" \
        -vf "select='not(mod(n\,7))',scale=640:360,tile=4x3" -frames:v 1 "${scene}sheet.png"
    done
    throttled=$(awk -F', ' '$2 > 50 && $1 != "P0"' target/playgate-gpu.csv | wc -l)
    if [ "$throttled" -gt 0 ]; then
      echo "the GPU worked below its full performance state $throttled times: nothing measured is to be kept"
      exit 1
    fi
    exit $status

# What the gate measures is worth nothing unless the machine has nothing else to do: with the
# gate built, so that building is not what is at work, the CPU and the GPU are watched for three
# seconds, and whatever is at work on them is named rather than measured along with the game.
idle:
    cargo test --release -p game --test play_gate --no-run
    cpu=$(vmstat 1 4 | tail -3 | awk '{idle += $15} END {print int(idle / 3)}'); \
    gpu=$(for _ in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 1; done \
      | awk '{busy += $1} END {print int(busy / 3)}'); \
    echo "CPU ${cpu}% idle, GPU ${gpu}% busy"; \
    if [ "$cpu" -lt 90 ] || [ "$gpu" -gt 25 ]; then \
      ps -eo pcpu,comm --sort=-pcpu | head -6; nvidia-smi pmon -c 1 -s u; \
      echo "the machine is not idle: the gate would measure what else is at work on it"; exit 1; \
    fi

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
