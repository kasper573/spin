import { BufferAttribute, BufferGeometry, Points, PointsMaterial } from 'three';

export function makeStars(count = 1600): Points {
  const pos = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const u = Math.random() * 2 - 1,
      t = Math.random() * Math.PI * 2;
    const r = Math.sqrt(1 - u * u),
      d = 120 + Math.random() * 60;
    pos[i * 3] = r * Math.cos(t) * d;
    pos[i * 3 + 1] = u * d;
    pos[i * 3 + 2] = r * Math.sin(t) * d;
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(pos, 3));
  const material = new PointsMaterial({
    color: 0xdde6ff,
    size: 0.55,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.8,
  });
  return new Points(geometry, material);
}
