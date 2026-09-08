// Camera-facing quad per fluid particle; the fragment shader carves a sphere out of it.
attribute vec3 ipos;
attribute float ifoam;
uniform float radius;
varying vec2 vLocal;
varying vec3 vCenter;
varying float vFoam;

void main() {
  vec4 c = modelViewMatrix * vec4(ipos, 1.0);
  vCenter = c.xyz;
  vFoam = ifoam;
  vLocal = position.xy;
  gl_Position = projectionMatrix * vec4(c.xyz + vec3(position.xy * radius, 0.0), 1.0);
}
