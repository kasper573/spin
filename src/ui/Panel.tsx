import type { JSX } from 'solid-js';
import { SPIN_MAX, SPIN_STEP, setSettings, settings, type Tool } from '../app/settings';
import type { Simulation } from '../app/simulation';
import { Gauge } from './Gauge';
import { Segmented, Slider, Toggle } from './controls';

const TOOLS: ReadonlyArray<{ value: Tool; label: string }> = [
  { value: 'inject', label: 'Inject water' },
  { value: 'drain', label: 'Drain water' },
];

export function Panel(props: { sim: () => Simulation | undefined }): JSX.Element {
  const nudgeSpin = (d: number) =>
    setSettings('spin', Math.max(0, Math.min(SPIN_MAX, settings.spin + d)));
  return (
    <div class="panel">
      <h1>Spin-gravity wheel</h1>
      <p class="sub">A solid glass drum in free fall. Spin it and watch inertia become a floor.</p>
      <Gauge />

      <section>
        <h2>Spin</h2>
        <div class="spin">
          <button type="button" title="Slow down" onClick={() => nudgeSpin(-SPIN_STEP)}>
            −
          </button>
          <input
            type="range"
            min={0}
            max={SPIN_MAX}
            step={0.05}
            value={settings.spin}
            aria-label="Target spin rate"
            onInput={(e) => setSettings('spin', parseFloat(e.currentTarget.value))}
          />
          <button type="button" title="Speed up" onClick={() => nudgeSpin(SPIN_STEP)}>
            +
          </button>
        </div>
        <div class="row">
          <label>Target rate</label>
          <output>{settings.spin.toFixed(2)} rad/s</output>
        </div>
      </section>

      <section>
        <h2>Water and rafts</h2>
        <div class="row">
          <label>Left button</label>
        </div>
        <Segmented
          label="Left mouse button"
          options={TOOLS}
          value={settings.tool}
          onChange={(t) => setSettings('tool', t)}
        />
        <Slider
          label="Flow rate"
          min={10}
          max={200}
          step={10}
          value={settings.flow}
          format={(v) => `${v} /s`}
          onInput={(v) => setSettings('flow', v)}
        />
        <Toggle
          label="Water and rafts enter moving with the wheel"
          checked={settings.matchWheel}
          onChange={(v) => setSettings('matchWheel', v)}
        />
        <div class="btns">
          <button type="button" class="warn" onClick={() => props.sim()?.clearRafts()}>
            Remove rafts
          </button>
          <button type="button" class="warn" onClick={() => props.sim()?.clearWater()}>
            Remove all water
          </button>
        </div>
      </section>

      <section>
        <h2>Physics</h2>
        <Slider
          label="Water viscosity"
          min={0}
          max={0.6}
          step={0.01}
          value={settings.viscosity}
          format={(v) => v.toFixed(2)}
          onInput={(v) => setSettings('viscosity', v)}
        />
        <Slider
          label="Glass–water drag"
          min={0}
          max={1}
          step={0.05}
          value={settings.wallFriction}
          format={(v) => v.toFixed(2)}
          onInput={(v) => setSettings('wallFriction', v)}
        />
        <Slider
          label="Raft friction μ"
          min={0}
          max={1}
          step={0.05}
          value={settings.raftFriction}
          format={(v) => v.toFixed(2)}
          onInput={(v) => setSettings('raftFriction', v)}
        />
        <Toggle
          label="Air inside the wheel (slowly drags free objects into rotation)"
          checked={settings.air}
          onChange={(v) => setSettings('air', v)}
        />
      </section>

      <section>
        <h2>Simulation</h2>
        <div class="btns">
          <button type="button" onClick={() => setSettings('paused', !settings.paused)}>
            {settings.paused ? 'Resume' : 'Pause'}
          </button>
          <button type="button" class="warn" onClick={() => props.sim()?.reset()}>
            Reset
          </button>
        </div>
      </section>
    </div>
  );
}
