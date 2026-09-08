import { createSignal, onCleanup, onMount, type JSX } from 'solid-js';
import { setSettings } from '../app/settings';
import { Simulation } from '../app/simulation';
import { Panel } from './Panel';
import { Hint, Status } from './Status';

export function App(): JSX.Element {
  const [sim, setSim] = createSignal<Simulation>();
  let canvas: HTMLCanvasElement | undefined;

  onMount(() => {
    if (!canvas) return;
    const s = new Simulation(canvas);
    s.start();
    setSim(s);
    if (import.meta.env.DEV) Object.assign(window, { sim: s, setSettings });
    onCleanup(() => s.dispose());
  });

  return (
    <>
      <canvas id="view" ref={(el) => (canvas = el)} />
      <Panel sim={sim} />
      <Hint />
      <Status />
    </>
  );
}
