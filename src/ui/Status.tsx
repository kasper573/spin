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
      <Show
        when={readouts.controlsActive}
        fallback={
          <>
            <b>Click the view</b> to take the controls.
          </>
        }
      >
        <b>WASD</b> fly, <b>Q E</b> roll, <b>R F</b> up/down, <b>mouse</b> look, <b>scroll</b>{' '}
        speed. <b>LMB</b> water, <b>RMB</b> raft, <b>Esc</b> release.
      </Show>
    </div>
  );
}

export function Crosshair(): JSX.Element {
  return <div class="crosshair" classList={{ active: readouts.controlsActive }} />;
}
