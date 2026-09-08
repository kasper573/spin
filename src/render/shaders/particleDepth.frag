// Writes the eye-space distance of the sphere surface, depth-tested against the opaque scene.
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
  float nz = sqrt(1.0 - r2);
  vec3 p = vCenter + vec3(vLocal * radius, nz * radius);
  vec4 clip = uProj * vec4(p, 1.0);
  float wd = clip.z / clip.w * 0.5 + 0.5;
  float sd = texture2D(tSceneDepth, gl_FragCoord.xy * invRes).r;
  if (wd > sd) discard;
  gl_FragDepth = wd;
  gl_FragColor = vec4(-p.z, 0.0, 0.0, 1.0);
}
