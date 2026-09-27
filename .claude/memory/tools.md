# Tools

Paths below are on the human's machine (RTX 3090, NVIDIA driver, X display `:1`).

## One gate scene

`cargo test -p game --release --test play_gate --no-run`, then run the printed binary from `game/`
with `--ignored --test-threads=1 <scene fn substring>` (after the clean-measurement steps in
workflow.md). Make its sheet with the ffmpeg line in the Justfile. To find frame-to-frame jumps,
keep every frame (temporarily set `KEPT_EVERY`, at 1080p) and diff consecutive frames; restore the
consts after. To debug the sheet, temporarily print the `held` rows.

## Nsight Systems

No root needed for GPU traces. CPU sampling needs `kernel.perf_event_paranoid=1` and
`kptr_restrict=0`, which the human once set but which read 4 after a reboot. Binary:
`~/.local/opt/nsight-systems/opt/nvidia/nsight-systems/2026.3.2/target-linux-x64/nsys`.

- GPU: from `game/`: `nsys profile --trace=vulkan --sample=none --cpuctxsw=none -o <out> <test
  binary> --ignored --test-threads=1 <scene>`, then `nsys export --type sqlite`: `VULKAN_WORKLOAD`
  = true GPU time per command buffer, mapped to its `vkQueueSubmit` by `correlationId`.
- GPU time per pass: run with `WGPU_DEBUG=1` (the headless device takes wgpu's flags from the
  environment) and `--trace=vulkan,vulkan-annotations`: `VULKAN_WORKLOAD` rows named
  `vkCmdBeginDebugUtilsLabelEXT` then carry each labelled pass's GPU time in `textId`
  (`main_transmissive_pass_3d`, `sheet`, `shadow_directional_light_0_cascade_N`, …). The
  transmission snapshots' texture copies sit outside the labels. Bevy's own pass timers miss
  about half of a view's GPU time: use these.
- Under nsys every pipeline compiles cold (~25 s in all, the driver's shader cache is not used),
  so a trace's first seconds say nothing about hitches in the gate.
- CPU spans without sampling: a `tracing` layer set as `LogPlugin.custom_layer` in
  `build_headless`, built with `--features bevy/trace,bevy/debug`, summing each span's time by
  thread.
- CPU sampling: build with `RUSTFLAGS="-C force-frame-pointers=yes" CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`,
  run with `DEBUGINFOD_URLS=` empty (otherwise it hangs), `--trace=none --sample=process-tree
  --backtrace=fp`, `nsys export --type sqlite`, aggregate `SAMPLING_CALLCHAINS` x
  `COMPOSITE_EVENTS` in python under `testing::frame`. Systems are inlined into `FnMut::call_mut`:
  name them by reading Bevy's source for the callee.
- `bevy/trace_chrome` works but writes ~15 GB per scene.
- Web-research subagents have reported wrong claims: treat them as leads only.

## Where frame time goes

With pipelined rendering the main thread and the render thread overlap, and `queue_submit` on the
render thread blocks once the GPU falls behind: a long `queue_submit` means GPU-bound. The portal
scene is GPU-bound (~24 ms of GPU a frame at 4K): the player's transmissive pass (the water) is
the largest single pass, then each portal eye's passes. Each see-through item a view draws costs a
copy of the whole 4K view texture first (Bevy's transmission snapshots), eyes included, though
they draw only a window of it. wgpu makes a staging buffer per `write_buffer`: ~400
`vkCreateBuffer` a frame.

## DLSS (parked)

SDK v310.5.3 at `~/.local/opt/dlss-sdk` (Bevy 0.19.1 pins `dlss_wgpu` 4.0.0 = that version); clang
and libclang-dev installed. `bevy/dlss` compiles with `DLSS_SDK=$HOME/.local/opt/dlss-sdk
VULKAN_SDK=/usr`; `force_disable_dlss` mocks it on machines without the SDK. To use it: ship
`libnvidia-ngx-dlss.so.310.5.3` and NVIDIA's licence with the native binary; insert `DlssProjectId`
before `DefaultPlugins`; check `Option<Res<DlssSuperResolutionSupported>>`; `Dlss` on the camera
needs `TemporalJitter`, `MipBias`, `DepthPrepass`, `MotionVectorPrepass`, `Hdr`; custom shaders
must honour `MainPassResolutionOverride`; the headless `testing::device()` needs a
`RawVulkanInitSettings`; the water meshes pull vertices from buffers, so their motion vectors need
checking. Bevy's TAA does not upscale, and there is no FSR.

## e2e (`e2e/smoke.mjs`)

Headless Chrome on the real GPU (SwiftShader takes ~45 s a frame and loses the device), ~4–12
fps. It waits on a same-origin `/blank` page until WebGPU offers an adapter. It restarts Chrome
on the same profile rather than reloading, because NVIDIA Xid 32 faults Chrome's GPU process on
its third device; the server port (origin) is kept within a run so storage survives. Probes kill
their browser in `finally`: a leaked headless Chrome spoils a gate run.

## Software rendering (machines without a GPU)

Mesa's lavapipe (`mesa-vulkan-drivers`; building also needs `libasound2-dev`, `libudev-dev`,
`pkg-config`) runs the gate with no code change: wgpu picks llvmpipe on its own. Frames are
stepped by simulated time, so they show the same game, only slowly; timings mean nothing about the
real GPU. On a 4-core cloud container the trickle scene ran at 218 ms a frame at 320x180 (~4 min
for its 14 s) and still 151 ms at 32x18, with the empty ring at 110 ms: a fixed floor of bloom
(~34 ms, its mip chain does not shrink with the view), the transmissive pass's 64 snapshots
(~27 ms), the idle particle solver's surface and bin passes, the sheet (~7 ms) and Bevy's CPU work.
No resolution reaches 60 fps there. To look at a scene: set the gate's `WIDTH`/`HEIGHT` to
320x180, run it as in "One gate scene", restore the consts.
