// Cel-style glass: fresnel-weighted tint, a soft pane sheen, a hard sun glint and a banded sky reflection.
uniform vec3 lightDir;
uniform vec3 fillDir;
uniform vec3 tint;
uniform mat3 uViewToWorld;
varying vec3 vN;
varying vec3 vP;

void main() {
  vec3 n = normalize(vN);
  if (!gl_FrontFacing) n = -n;
  vec3 v = normalize(-vP);
  float ndv = max(dot(n, v), 0.0);
  float fres = pow(1.0 - ndv, 3.0);

  vec3 h = normalize(lightDir + v);
  float ndh = max(dot(n, h), 0.0);
  float glint = smoothstep(0.986, 0.992, ndh) * 0.9;
  float sheen = pow(ndh, 10.0) * 0.12 + pow(max(dot(n, normalize(fillDir + v)), 0.0), 10.0) * 0.08;

  vec3 rw = uViewToWorld * reflect(-v, n);
  float sky = smoothstep(0.30, 0.42, rw.y) * 0.10;

  vec3 col = tint * (0.35 + 0.65 * fres) + vec3(sheen + sky + glint);
  float alpha = 0.08 + 0.45 * fres + sheen + sky + glint * 0.7;
  if (!gl_FrontFacing) alpha *= 0.6;
  gl_FragColor = linearToOutputTexel(vec4(col, clamp(alpha, 0.0, 1.0)));
}
