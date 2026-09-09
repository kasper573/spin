// Loads the built page in headless Chrome, drives it through the script hooks the client exposes
// and checks the simulation and its persistence behave. Run via `just e2e`.
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { connect, launchChrome, sleep } from "./cdp.mjs";

async function waitFor(condition, what, seconds = 60) {
  for (let attempt = 0; attempt < seconds * 10; attempt++) {
    if (await condition()) return;
    await sleep(100);
  }
  throw new Error(`timed out waiting for ${what}`);
}

// Software rendering on a CI runner manages a frame every few seconds, each a fraction of a second
// of simulation, so an advance is only given up on when the simulation stops making progress.
async function waitForProgress(measure, target, what, patience = 60) {
  let last = await measure();
  let stalled = 0;
  while (last < target) {
    await sleep(100);
    const now = await measure();
    stalled = now > last ? 0 : stalled + 1;
    if (stalled >= patience * 10) throw new Error(`timed out waiting for ${what}`);
    last = now;
  }
}

const dist = path.resolve(import.meta.dirname, "..", "dist");
const out = path.resolve(import.meta.dirname, "..", "target", "e2e");
fs.mkdirSync(out, { recursive: true });

const server = http
  .createServer((req, res) => {
    const file = path.join(dist, req.url === "/" ? "index.html" : req.url.split("?")[0]);
    const types = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm" };
    fs.readFile(file, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end();
        return;
      }
      res.writeHead(200, { "Content-Type": types[path.extname(file)] ?? "application/octet-stream" });
      res.end(data);
    });
  })
  .listen(0, "127.0.0.1");
await new Promise((resolve) => server.once("listening", resolve));
const port = server.address().port;
const url = `http://127.0.0.1:${port}/`;

const { chrome, page } = await launchChrome(9333, path.join(out, "chrome-profile"));
const { send, evaluate, logs, ready } = connect(page);
await ready;
await send("Runtime.enable");
await send("Log.enable");
await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 800, height: 500, deviceScaleFactor: 1, mobile: false });

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` (${detail})` : ""}`);
  if (!ok) failures++;
};
const status = async () => JSON.parse(await evaluate("window.spin_status()"));
// Commands run on a later frame and the status is published at the end of each frame, so wait
// for two more frames before reading anything back.
const command = async (cmd) => {
  const before = await status();
  await evaluate(`window.spin_command(${JSON.stringify(JSON.stringify(cmd))})`);
  await waitFor(async () => (await status()).frame >= before.frame + 2, `frames after ${cmd.cmd}`);
  if (cmd.cmd === "advance") {
    await waitForProgress(async () => (await status()).time, before.time + cmd.seconds - 1e-3, `the simulation to advance ${cmd.seconds} s`);
  }
};
const screenshot = async (name) => {
  const r = await send("Page.captureScreenshot", { format: "png" });
  fs.writeFileSync(path.join(out, `${name}.png`), Buffer.from(r.data, "base64"));
};
const waitForApp = async () => {
  for (let i = 0; i < 300; i++) {
    const ok = await evaluate("typeof window.spin_status === 'function' && window.spin_status().length > 0");
    if (ok) return true;
    await sleep(200);
  }
  return false;
};

try {
  await evaluate("localStorage.clear(); 1").catch(() => {});
  await send("Page.navigate", { url });
  check("app starts", await waitForApp());
  const fresh = await status();
  check("starts empty", fresh.particles === 0, JSON.stringify(fresh));

  check("stands on the ground under one g", Math.abs(fresh.weight - 1) < 0.05 && fresh.ground_speed < 0.1, `${fresh.weight} g, ${fresh.ground_speed} m/s`);
  await screenshot("standing");

  await command({ cmd: "thrust", forward: 1, seconds: 4 });
  await command({ cmd: "advance", seconds: 2 });
  const walking = await status();
  check("forward thrust walks at walking pace", walking.ground_speed > 1.2 && walking.ground_speed < 1.8, `${walking.ground_speed} m/s`);
  check("the forward thruster reads full", walking.thrust[0] === 1 && walking.thrust.slice(1).every((l) => l === 0), JSON.stringify(walking.thrust));
  await command({ cmd: "advance", seconds: 3 });
  await command({ cmd: "thrust", up: 1, seconds: 0.8 });
  await command({ cmd: "advance", seconds: 1.0 });
  const jumping = await status();
  check("up thrust lifts off the ground", jumping.airborne && jumping.weight === 0, JSON.stringify({ airborne: jumping.airborne, weight: jumping.weight }));
  await command({ cmd: "advance", seconds: 3 });
  const landed = await status();
  check("and lands again", !landed.airborne && Math.abs(landed.weight - 1) < 0.1 && landed.ground_speed < 0.2, JSON.stringify({ airborne: landed.airborne, weight: landed.weight, speed: landed.ground_speed }));

  await command({ cmd: "spin", value: 1.0 });
  await command({ cmd: "advance", seconds: 2 });
  for (let k = 0; k < 12; k++) {
    const a = k * 0.52;
    await command({ cmd: "inject", x: Math.cos(a) * 8.0, y: (k % 3) * 2.0 - 2.0, z: Math.sin(a) * 8.0, count: 150 });
    await command({ cmd: "advance", seconds: 0.2 });
  }
  await command({ cmd: "advance", seconds: 8 });
  const filled = await status();
  check("water injected", filled.particles === 1800, `${filled.particles} particles, ${filled.litres} L`);
  check("drum spinning", Math.abs(filled.spin - 1.0) < 1e-3, `${filled.spin}`);
  await command({ cmd: "camera", x: 14.7, y: 16.5, z: 21.6, look_x: 0, look_y: 0, look_z: 0 });
  await screenshot("water");

  await command({ cmd: "sculpt", phi: 0.8, y: 0, radius: 3.0, amount: 1.0 });
  await command({ cmd: "advance", seconds: 3 });
  const land = await status();
  check("landscape raised", land.landscape_max > 1.2, `${land.landscape_max}`);
  await command({ cmd: "spin", value: 0 });
  await command({ cmd: "advance", seconds: 4 });
  const still = await status();
  check("drum stopped", still.spin === 0, `${still.spin}`);
  const a = 0.8 - still.angle;
  await command({ cmd: "camera", x: Math.cos(a) * 3.0, y: 2.5, z: Math.sin(a) * 3.0, look_x: Math.cos(a) * 9.0, look_y: 0, look_z: Math.sin(a) * 9.0 });
  await screenshot("landscape-inside");
  await command({ cmd: "camera", x: Math.cos(a) * 16.0, y: 7.0, z: Math.sin(a) * 16.0, look_x: Math.cos(a) * 9.0, look_y: 0, look_z: Math.sin(a) * 9.0 });
  await screenshot("landscape-outside");
  await command({ cmd: "camera", x: 14.7, y: 16.5, z: 21.6, look_x: 0, look_y: 0, look_z: 0 });
  await command({ cmd: "spin", value: 1.0 });
  await command({ cmd: "advance", seconds: 4 });

  await command({ cmd: "ring", diameter: 30, width: 16 });
  await command({ cmd: "advance", seconds: 2 });
  const resized = await status();
  check("the ring resizes and keeps its water", resized.diameter === 30 && resized.width === 16 && resized.particles === 1800 && resized.landscape_max > 1.2, JSON.stringify({ diameter: resized.diameter, width: resized.width, particles: resized.particles, land: resized.landscape_max }));
  await command({ cmd: "equalize" });
  await command({ cmd: "advance", seconds: 0.2 });
  const equalized = await status();
  check("thruster power equalizes to the bigger ring's gravity", equalized.thrust_power > 1.78 * 14.0 && equalized.thrust_power < 1.78 * 14.6, `${equalized.thrust_power} m/s2`);
  await screenshot("resized");
  await command({ cmd: "ring", diameter: 21, width: 12 });
  await command({ cmd: "advance", seconds: 2 });

  await sleep(2000);
  const live = await status();
  check("frames render", live.fps > 1, `${live.fps.toFixed(1)} fps, sim ${(live.sim_rate * 100).toFixed(0)}%`);

  const before = (await status()).saves;
  await command({ cmd: "save" });
  await waitFor(async () => (await status()).saves > before, "the save to land");
  await send("Page.reload");
  check("app restarts", await waitForApp());
  const restored = await status();
  check("state persists across reload", restored.particles === live.particles && restored.landscape_max > 1.2 && Math.abs(restored.spin - 1.0) < 1e-3, JSON.stringify({ particles: restored.particles, land: restored.landscape_max, spin: restored.spin }));
  await screenshot("restored");

  const errors = [...new Set(logs.filter((l) => l.startsWith("[exception]") || l.includes("panicked") || l.startsWith("[log:error]")))];
  check("no errors in the console", errors.length === 0, errors.join(" | ").slice(0, 1500));
} catch (error) {
  failures++;
  console.log("FAIL", error.stack ?? error);
  console.log(logs.slice(-20).join("\n"));
} finally {
  chrome.kill();
  server.close();
}
console.log(failures ? `${failures} check(s) failed` : "e2e passed");
process.exit(failures ? 1 : 0);
