// Minimal Chrome DevTools Protocol client over the browser's own WebSocket: no dependencies.
import { spawn } from "node:child_process";

export async function launchChrome(port, profileDir) {
  const binary = process.env.CHROME_BIN ?? "google-chrome";
  const chrome = spawn(
    binary,
    [
      "--headless=new",
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${profileDir}`,
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
      "--enable-unsafe-webgpu",
      "--use-vulkan=swiftshader",
      "--enable-features=Vulkan",
      "--no-sandbox",
      "--disable-dev-shm-usage",
      "--window-size=1400,900",
      "about:blank",
    ],
    { stdio: "ignore" },
  );
  // A cold start on a software-rendered CI runner can take well over ten seconds.
  for (let attempt = 0; attempt < 600; attempt++) {
    if (chrome.exitCode !== null) throw new Error(`chrome exited with code ${chrome.exitCode}`);
    try {
      const pages = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
      const page = pages.find((p) => p.type === "page");
      if (page) return { chrome, page };
    } catch {}
    await sleep(100);
  }
  chrome.kill();
  throw new Error("chrome did not start within 60 s");
}

export function connect(page) {
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 0;
  const pending = new Map();
  const logs = [];
  ws.onmessage = (event) => {
    const m = JSON.parse(event.data);
    if (m.id && pending.has(m.id)) {
      pending.get(m.id)(m.result ?? m.error);
      pending.delete(m.id);
    } else if (m.method === "Runtime.consoleAPICalled") {
      logs.push(`[${m.params.type}] ` + m.params.args.map((a) => a.value ?? a.description ?? "").join(" "));
    } else if (m.method === "Runtime.exceptionThrown") {
      logs.push("[exception] " + (m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text));
    } else if (m.method === "Log.entryAdded") {
      logs.push(`[log:${m.params.entry.level}] ${m.params.entry.text}`);
    }
  };
  const send = (method, params = {}) =>
    new Promise((resolve) => {
      const i = ++id;
      pending.set(i, resolve);
      ws.send(JSON.stringify({ id: i, method, params }));
    });
  const evaluate = async (expression) => {
    const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? "evaluate failed");
    return r.result?.value;
  };
  const ready = new Promise((resolve) => (ws.onopen = resolve));
  return { ws, send, evaluate, logs, ready };
}

export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
