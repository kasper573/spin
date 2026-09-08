// Shades the smoothed fluid surface: cel bands, refraction of the scene behind it, absorption by
// thickness, foam from particle agitation, fresnel rim, stepped specular and an inked outline.
uniform sampler2D tScene;
uniform sampler2D tSceneDepth;
uniform sampler2D tDepth;
uniform sampler2D tThick;
uniform vec2 invRes;
uniform vec2 invP;
uniform mat4 uProj;
uniform mat4 uViewInv;
uniform vec3 lightDir;
uniform float time;
uniform float theta;
varying vec2 vUv;

vec3 eyePos(vec2 uv, float d) {
  return vec3((uv * 2.0 - 1.0) * invP * d, -d);
}

float hash(vec3 p) {
  p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x), mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
    mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x), mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y),
    f.z);
}

// One-sided difference toward whichever neighbour is closer in depth, so creases stay sharp.
vec3 tangent(vec2 uv, vec2 o, vec3 p, float d, float dp, float dm, vec3 fallback) {
  bool okP = dp > 0.0, okM = dm > 0.0;
  if (okP && okM) return abs(dp - d) < abs(dm - d) ? eyePos(uv + o, dp) - p : p - eyePos(uv - o, dm);
  if (okP) return eyePos(uv + o, dp) - p;
  if (okM) return p - eyePos(uv - o, dm);
  return fallback;
}

void main() {
  vec4 sceneC = texture2D(tScene, vUv);
  float sceneD = texture2D(tSceneDepth, vUv).r;
  float d = texture2D(tDepth, vUv).r;
  if (d <= 0.0) {
    gl_FragColor = linearToOutputTexel(sceneC);
    gl_FragDepth = sceneD;
    return;
  }

  vec2 ox = vec2(invRes.x, 0.0), oy = vec2(0.0, invRes.y);
  float dr = texture2D(tDepth, vUv + ox).r, dl = texture2D(tDepth, vUv - ox).r;
  float du = texture2D(tDepth, vUv + oy).r, dn = texture2D(tDepth, vUv - oy).r;
  vec3 p = eyePos(vUv, d);
  vec3 ddx = tangent(vUv, ox, p, d, dr, dl, vec3(2.0 * invRes.x * invP.x * d, 0.0, 0.0));
  vec3 ddy = tangent(vUv, oy, p, d, du, dn, vec3(0.0, 2.0 * invRes.y * invP.y * d, 0.0));
  vec3 n = normalize(cross(ddx, ddy));
  vec3 v = normalize(-p);
  float ndv = max(dot(n, v), 0.0);
  float ndl = dot(n, lightDir);

  vec4 th = texture2D(tThick, vUv);
  float T = th.r;
  float foamAvg = th.g / max(T, 1e-4);

  // three-tone cel base
  float band = smoothstep(-0.05, 0.05, ndl - 0.02) * 0.5 + smoothstep(-0.05, 0.05, ndl - 0.55) * 0.5;
  vec3 deep = vec3(0.02, 0.17, 0.48), mid = vec3(0.07, 0.42, 0.84), light = vec3(0.32, 0.70, 0.98);
  vec3 cel = mix(deep, mid, smoothstep(0.0, 0.5, band));
  cel = mix(cel, light, smoothstep(0.5, 1.0, band));

  // what lies behind, refracted and absorbed by the water thickness
  vec4 clip = uProj * vec4(p, 1.0);
  float wd = clip.z / clip.w * 0.5 + 0.5;
  vec2 ruv = vUv + n.xy * 0.05 * clamp(T, 0.0, 1.5);
  if (texture2D(tSceneDepth, ruv).r < wd) ruv = vUv;
  vec3 behind = texture2D(tScene, ruv).rgb;
  vec3 absorb = exp(-T * vec3(1.6, 0.7, 0.35));
  float opaq = 0.35 + 0.65 * (1.0 - exp(-T * 0.8));
  vec3 col = mix(behind * absorb, cel, opaq);

  // thin sheets and flying droplets are paler and more translucent
  float thin = 1.0 - smoothstep(0.0, 0.4, T);
  col = mix(col, vec3(0.55, 0.85, 1.0), thin * 0.3);

  // foam: agitation from the simulation, broken up by noise that co-rotates with the drum
  vec3 wp = (uViewInv * vec4(p, 1.0)).xyz;
  float c = cos(theta), s = sin(theta);
  vec3 pr = vec3(c * wp.x - s * wp.z, wp.y, s * wp.x + c * wp.z);
  float n1 = noise(pr * 7.0 + vec3(0.0, time * 0.4, 0.0));
  float n2 = noise(pr * 11.0 - vec3(time * 0.7));
  float f = foamAvg * 1.2 + 0.26 * n1 + 0.12 * n2 - 0.28;
  float foamMask = smoothstep(0.45, 0.58, f);
  col = mix(col, vec3(0.95, 0.98, 1.0), foamMask * 0.85);

  // fresnel rim and a hard specular glint
  float fres = pow(1.0 - ndv, 4.0);
  col += vec3(0.5, 0.75, 1.0) * fres * 0.45 * (1.0 - foamMask);
  vec3 h = normalize(lightDir + v);
  col += vec3(1.0) * smoothstep(0.975, 0.985, dot(n, h)) * 0.85;

  // inked outline: silhouettes, depth creases and grazing angles
  vec2 ox2 = 2.0 * ox, oy2 = 2.0 * oy;
  float e0 = texture2D(tDepth, vUv + ox2).r, e1 = texture2D(tDepth, vUv - ox2).r;
  float e2 = texture2D(tDepth, vUv + oy2).r, e3 = texture2D(tDepth, vUv - oy2).r;
  float ink = smoothstep(0.12, 0.30, max(max(abs(dr - d), abs(dl - d)), max(abs(du - d), abs(dn - d))));
  if (e0 <= 0.0 || e1 <= 0.0 || e2 <= 0.0 || e3 <= 0.0) ink = 1.0;
  ink = max(ink, smoothstep(0.16, 0.02, ndv));
  col *= 1.0 - 0.35 * ink;

  float alpha = smoothstep(0.0, 0.05, T);
  gl_FragColor = linearToOutputTexel(vec4(mix(sceneC.rgb, col, alpha), 1.0));
  gl_FragDepth = wd;
}
