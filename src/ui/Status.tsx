import { Show, type JSX } from 'solid-js';
import { MASS } from '../physics/constants';
import { readouts } from '../app/settings';

export function Status(): JSX.Element {
  return (
    <div class="status">
      About {Math.round(readouts.particles * MASS)} L of water. {readouts.rafts} raft
      {readouts.rafts === 1 ? '' : 's'}. {Math.round(readouts.fps)} fps
      <Show when={readouts.simRate < 0.95}>
        , sim running at {Math.round(readouts.simRate * 100)}% speed
      </Show>
    </div>
  );
}

export function Hint(): JSX.Element {
  return (
    <div class="hint">
      <b>Left drag</b> uses the tool. <b>Right drag</b> or Shift orbits. <b>Scroll</b> zooms.
    </div>
  );
}
