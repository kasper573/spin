varying vec3 vN;
varying vec3 vP;

void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalize(normalMatrix * normal);
  vP = mv.xyz;
  gl_Position = projectionMatrix * mv;
}
