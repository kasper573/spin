import type { JSX } from 'solid-js';
import { R_OUT } from '../physics/constants';
import { readouts } from '../app/settings';

const G = 9.81;
const HEAD_HEIGHT = 1.8;
const BAR_FULL_G = 3.2;

/** Artificial gravity readout derived from the current spin. */
export function Gauge(): JSX.Element {
  const floor = () => readouts.omega * readouts.omega * R_OUT;
  const head = () => readouts.omega * readouts.omega * Math.max(0, R_OUT - HEAD_HEIGHT);
  const rpm = () => (readouts.omega * 60) / (2 * Math.PI);
  return (
    <>
      <div class="gauge">
        <span class="big">{(floor() / G).toFixed(2)}</span>
        <span class="unit">g at the floor</span>
      </div>
      <div class="gbar">
        <i style={{ width: `${Math.min(100, (floor() / G / BAR_FULL_G) * 100)}%` }} />
      </div>
      <div class="fine">
        <span>
          Spin <b>{rpm().toFixed(1)} rpm</b>
        </span>
        <span>
          Floor speed <b>{(readouts.omega * R_OUT).toFixed(1)} m/s</b>
        </span>
        <span>
          At head height <b>{(head() / G).toFixed(2)} g</b>
        </span>
        <span>
          Floor <b>{floor().toFixed(1)} m/s²</b>
        </span>
      </div>
    </>
  );
}
