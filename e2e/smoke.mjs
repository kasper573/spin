// Loads the built page in headless Chrome, drives it through the script hooks the client exposes
// and checks the simulation and its persistence behave. Run via `just e2e`.
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { connect, launchChrome, sleep } from "./cdp.mjs";

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
await send("Emulation.setDeviceMetricsOverride", { width: 1400, height: 900, deviceScaleFactor: 1, mobile: false });

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` (${detail})` : ""}`);
  if (!ok) failures++;
};
const status = async () => JSON.parse(await evaluate("window.spin_status()"));
const command = (cmd) => evaluate(`window.spin_command(${JSON.stringify(JSON.stringify(cmd))})`);
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
  await sleep(1500);
  const fresh = await status();
  check("starts empty", fresh.particles === 0 && fresh.rafts === 0, JSON.stringify(fresh));

  await command({ cmd: "spin", value: 1.6 });
  await command({ cmd: "advance", seconds: 4 });
  for (let k = 0; k < 12; k++) {
    const a = k * 0.52;
    await command({ cmd: "inject", x: Math.cos(a) * 2.7, y: (k % 3) * 0.3 - 0.3, z: Math.sin(a) * 2.7, count: 150 });
    await command({ cmd: "advance", seconds: 0.2 });
  }
  for (let k = 0; k < 3; k++) {
    const a = k * 2.0;
    await command({ cmd: "raft", x: Math.cos(a) * 3.1, y: 0, z: Math.sin(a) * 3.1, nx: -Math.cos(a), ny: 0, nz: -Math.sin(a) });
  }
  await command({ cmd: "advance", seconds: 8 });
  await sleep(1500);
  const filled = await status();
  check("water injected", filled.particles === 1800, `${filled.particles} particles, ${filled.litres} L`);
  check("rafts placed", filled.rafts === 3, `${filled.rafts}`);
  check("rafts ride with the glass", filled.raft_slip.every(([r, slip]) => r > 2.6 && Math.abs(slip) < 1.0), JSON.stringify(filled.raft_slip));
  check("drum spinning", Math.abs(filled.spin - 1.6) < 1e-3, `${filled.spin}`);
  await screenshot("water");

  await command({ cmd: "sculpt", phi: 0.8, y: 0, radius: 1.2, amount: 1.0 });
  await command({ cmd: "advance", seconds: 3 });
  await sleep(800);
  const land = await status();
  check("landscape raised", land.landscape_max > 0.5, `${land.landscape_max}`);
  await command({ cmd: "spin", value: 0 });
  await command({ cmd: "advance", seconds: 4 });
  await sleep(1500);
  const still = await status();
  check("drum stopped", still.spin === 0, `${still.spin}`);
  const a = 0.8 - still.angle;
  await command({ cmd: "camera", x: Math.cos(a) * 0.8, y: 0.9, z: Math.sin(a) * 0.8, look_x: Math.cos(a) * 3.0, look_y: 0, look_z: Math.sin(a) * 3.0 });
  await sleep(800);
  await screenshot("landscape-inside");
  await command({ cmd: "camera", x: Math.cos(a) * 5.5, y: 2.5, z: Math.sin(a) * 5.5, look_x: Math.cos(a) * 3.0, look_y: 0, look_z: Math.sin(a) * 3.0 });
  await sleep(800);
  await screenshot("landscape-outside");
  await command({ cmd: "camera", x: 4.9, y: 5.5, z: 7.2, look_x: 0, look_y: 0, look_z: 0 });
  await command({ cmd: "spin", value: 1.6 });
  await command({ cmd: "advance", seconds: 4 });

  await sleep(2000);
  const live = await status();
  check("frames render", live.fps > 1, `${live.fps.toFixed(1)} fps, sim ${(live.sim_rate * 100).toFixed(0)}%`);

  await command({ cmd: "save" });
  await sleep(300);
  await send("Page.reload");
  check("app restarts", await waitForApp());
  await sleep(1500);
  const restored = await status();
  check("state persists across reload", restored.particles === live.particles && restored.rafts === 3 && restored.landscape_max > 0.5 && Math.abs(restored.spin - 1.6) < 1e-3, JSON.stringify({ particles: restored.particles, rafts: restored.rafts, land: restored.landscape_max, spin: restored.spin }));
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
