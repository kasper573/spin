// Additive thickness (R) and foam-weighted thickness (G) of the water along the view ray.
uniform sampler2D tSceneDepth;
uniform vec2 invRes;
uniform mat4 uProj;
uniform float radius;
varying vec2 vLocal;
varying vec3 vCenter;
varying float vFoam;

void main() {
  float r2 = dot(vLocal, vLocal);
  if (r2 > 1.0) discard;
  vec4 clip = uProj * vec4(vCenter, 1.0);
  float wd = clip.z / clip.w * 0.5 + 0.5;
  float sd = texture2D(tSceneDepth, gl_FragCoord.xy * invRes).r;
  if (wd > sd) discard;
  float t = (1.0 - r2) * (1.0 - r2) * radius;
  gl_FragColor = vec4(t, t * vFoam, 0.0, 1.0);
}
