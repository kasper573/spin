// Separable bilateral blur of the fluid depth buffer with a world-space kernel radius.
uniform sampler2D tDepth;
uniform vec2 dir;
uniform float projScale;
uniform float worldRadius;
uniform float rangeSigma;
varying vec2 vUv;

void main() {
  float d = texture2D(tDepth, vUv).r;
  if (d <= 0.0) {
    gl_FragColor = vec4(0.0);
    return;
  }
  float rad = clamp(worldRadius * projScale / d, 1.0, 20.0);
  float stp = rad / 10.0;
  float sig2 = 2.0 * (rad * 0.5) * (rad * 0.5);
  float sum = d, wsum = 1.0;
  for (int i = 1; i <= 10; i++) {
    float o = float(i) * stp;
    float g = exp(-(o * o) / sig2);
    float a = texture2D(tDepth, vUv + dir * o).r;
    if (a > 0.0) {
      float dz = (a - d) / rangeSigma;
      float w = g * exp(-0.5 * dz * dz);
      sum += a * w;
      wsum += w;
    }
    float b = texture2D(tDepth, vUv - dir * o).r;
    if (b > 0.0) {
      float dz = (b - d) / rangeSigma;
      float w = g * exp(-0.5 * dz * dz);
      sum += b * w;
      wsum += w;
    }
  }
  gl_FragColor = vec4(sum / wsum, 0.0, 0.0, 1.0);
}
