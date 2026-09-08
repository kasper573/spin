/* Screen-space fluid rendering (van der Laan et al. 2009): particles are splatted as spheres into a
   depth buffer that is smoothed with a bilateral filter, plus an additive thickness/foam buffer; a
   final full-screen pass reconstructs normals and shades the surface over the opaque scene. */
import {
  AddEquation,
  BufferAttribute,
  CustomBlending,
  FloatType,
  HalfFloatType,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  LinearFilter,
  Matrix4,
  Mesh,
  NearestFilter,
  OneFactor,
  OrthographicCamera,
  PlaneGeometry,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  Vector2,
  Vector3,
  WebGLRenderTarget,
  type PerspectiveCamera,
  type Texture,
  type TextureDataType,
  type WebGLRenderer,
  AlwaysDepth,
} from 'three';
import { D, MAXP } from '../physics/constants';
import type { Fluid } from '../physics/fluid';
import blurFrag from './shaders/blur.frag?raw';
import compositeFrag from './shaders/composite.frag?raw';
import fullscreenVert from './shaders/fullscreen.vert?raw';
import particleVert from './shaders/particle.vert?raw';
import particleDepthFrag from './shaders/particleDepth.frag?raw';
import particleThickFrag from './shaders/particleThick.frag?raw';

const DEPTH_RADIUS = 1.2 * D; // sphere radius used for the surface
const THICK_RADIUS = 1.7 * D; // wider footprint for a smooth thickness field
const BLUR_RADIUS = 1.6 * D; // world-space smoothing radius
const BLUR_RANGE = 1.25 * D; // depth difference (m) at which neighbours stop contributing
const BLUR_PASSES = 2;

function floatTargetType(renderer: WebGLRenderer): TextureDataType {
  const gl = renderer.getContext();
  return gl.getExtension('EXT_color_buffer_float') ? FloatType : HalfFloatType;
}

function makeTarget(
  w: number,
  h: number,
  type: TextureDataType,
  nearest: boolean,
  depth: boolean,
): WebGLRenderTarget {
  const f = nearest ? NearestFilter : LinearFilter;
  return new WebGLRenderTarget(w, h, {
    type,
    format: RGBAFormat,
    minFilter: f,
    magFilter: f,
    depthBuffer: depth,
    stencilBuffer: false,
    generateMipmaps: false,
  });
}

export class WaterPass {
  private readonly geometry: InstancedBufferGeometry;
  private readonly positions: InstancedBufferAttribute;
  private readonly foam: InstancedBufferAttribute;
  private readonly posBuf = new Float32Array(MAXP * 3);
  private readonly depthMat: ShaderMaterial;
  private readonly thickMat: ShaderMaterial;
  private readonly blurMat: ShaderMaterial;
  private readonly compositeMat: ShaderMaterial;
  private readonly particleScene = new Scene();
  private readonly depthMesh: Mesh;
  private readonly thickMesh: Mesh;
  private readonly quadScene = new Scene();
  private readonly quad: Mesh;
  private readonly ortho = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly type: TextureDataType;
  private depthA: WebGLRenderTarget;
  private depthB: WebGLRenderTarget;
  private thick: WebGLRenderTarget;
  private width = 1;
  private height = 1;

  constructor(private readonly renderer: WebGLRenderer) {
    this.type = floatTargetType(renderer);

    this.geometry = new InstancedBufferGeometry();
    this.geometry.setAttribute(
      'position',
      new BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]), 3),
    );
    this.geometry.setIndex([0, 1, 2, 0, 2, 3]);
    this.positions = new InstancedBufferAttribute(this.posBuf, 3);
    this.foam = new InstancedBufferAttribute(new Float32Array(MAXP), 1);
    this.geometry.setAttribute('ipos', this.positions);
    this.geometry.setAttribute('ifoam', this.foam);
    this.geometry.instanceCount = 0;

    const particleUniforms = () => ({
      tSceneDepth: { value: null as Texture | null },
      invRes: { value: new Vector2() },
      uProj: { value: new Matrix4() },
      radius: { value: DEPTH_RADIUS },
    });
    this.depthMat = new ShaderMaterial({
      uniforms: particleUniforms(),
      vertexShader: particleVert,
      fragmentShader: particleDepthFrag,
    });
    this.thickMat = new ShaderMaterial({
      uniforms: particleUniforms(),
      vertexShader: particleVert,
      fragmentShader: particleThickFrag,
      depthTest: false,
      depthWrite: false,
      transparent: true,
      blending: CustomBlending,
      blendEquation: AddEquation,
      blendSrc: OneFactor,
      blendDst: OneFactor,
    });
    this.thickMat.uniforms.radius.value = THICK_RADIUS;
    this.depthMesh = new Mesh(this.geometry, this.depthMat);
    this.thickMesh = new Mesh(this.geometry, this.thickMat);
    this.depthMesh.frustumCulled = this.thickMesh.frustumCulled = false;
    this.particleScene.add(this.depthMesh, this.thickMesh);

    this.blurMat = new ShaderMaterial({
      uniforms: {
        tDepth: { value: null },
        dir: { value: new Vector2() },
        projScale: { value: 1 },
        worldRadius: { value: BLUR_RADIUS },
        rangeSigma: { value: BLUR_RANGE },
      },
      vertexShader: fullscreenVert,
      fragmentShader: blurFrag,
      depthTest: false,
      depthWrite: false,
    });
    this.compositeMat = new ShaderMaterial({
      uniforms: {
        tScene: { value: null },
        tSceneDepth: { value: null },
        tDepth: { value: null },
        tThick: { value: null },
        invRes: { value: new Vector2() },
        invP: { value: new Vector2() },
        uProj: { value: new Matrix4() },
        uViewInv: { value: new Matrix4() },
        lightDir: { value: new Vector3() },
        time: { value: 0 },
        theta: { value: 0 },
      },
      vertexShader: fullscreenVert,
      fragmentShader: compositeFrag,
      depthTest: true,
      depthFunc: AlwaysDepth,
      depthWrite: true,
    });
    this.quad = new Mesh(new PlaneGeometry(2, 2), this.blurMat);
    this.quad.frustumCulled = false;
    this.quadScene.add(this.quad);

    this.depthA = makeTarget(1, 1, this.type, true, true);
    this.depthB = makeTarget(1, 1, this.type, true, false);
    this.thick = makeTarget(1, 1, this.type, false, false);
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.depthA.setSize(width, height);
    this.depthB.setSize(width, height);
    this.thick.setSize(width, height);
    const inv = new Vector2(1 / width, 1 / height);
    (this.depthMat.uniforms.invRes.value as Vector2).copy(inv);
    (this.thickMat.uniforms.invRes.value as Vector2).copy(inv);
    (this.compositeMat.uniforms.invRes.value as Vector2).copy(inv);
  }

  /** Copy particle state into the instance buffers. */
  upload(F: Fluid): void {
    const n = F.n,
      p = this.posBuf;
    for (let i = 0; i < n; i++) {
      p[i * 3] = F.x[i];
      p[i * 3 + 1] = F.y[i];
      p[i * 3 + 2] = F.z[i];
    }
    (this.foam.array as Float32Array).set(F.foam.subarray(0, n));
    this.positions.needsUpdate = true;
    this.foam.needsUpdate = true;
    this.geometry.instanceCount = n;
  }

  /**
   * Draws the water over `sceneTarget` into the currently bound target (the screen), writing depth so
   * that later passes can be depth-tested against water and scene alike.
   */
  render(
    camera: PerspectiveCamera,
    sceneTarget: WebGLRenderTarget,
    sunDirWorld: Vector3,
    time: number,
    theta: number,
  ): void {
    const r = this.renderer;
    const sceneDepth = sceneTarget.depthTexture;
    const hasWater = this.geometry.instanceCount > 0;
    const proj = camera.projectionMatrix;

    for (const m of [this.depthMat, this.thickMat]) {
      m.uniforms.tSceneDepth.value = sceneDepth;
      (m.uniforms.uProj.value as Matrix4).copy(proj);
    }

    r.setClearColor(0x000000, 0);
    r.setRenderTarget(this.depthA);
    r.clear(true, true, false);
    if (hasWater) {
      this.depthMesh.visible = true;
      this.thickMesh.visible = false;
      r.render(this.particleScene, camera);
    }

    r.setRenderTarget(this.thick);
    r.clear(true, false, false);
    if (hasWater) {
      this.depthMesh.visible = false;
      this.thickMesh.visible = true;
      r.render(this.particleScene, camera);
    }

    let src = this.depthA,
      dst = this.depthB;
    if (hasWater) {
      const bu = this.blurMat.uniforms;
      bu.projScale.value = this.height / (2 * Math.tan((camera.fov * Math.PI) / 360));
      this.quad.material = this.blurMat;
      for (let pass = 0; pass < BLUR_PASSES * 2; pass++) {
        bu.tDepth.value = src.texture;
        (bu.dir.value as Vector2).set(
          pass % 2 === 0 ? 1 / this.width : 0,
          pass % 2 === 0 ? 0 : 1 / this.height,
        );
        r.setRenderTarget(dst);
        r.render(this.quadScene, this.ortho);
        [src, dst] = [dst, src];
      }
    }

    const cu = this.compositeMat.uniforms;
    cu.tScene.value = sceneTarget.texture;
    cu.tSceneDepth.value = sceneDepth;
    cu.tDepth.value = src.texture;
    cu.tThick.value = this.thick.texture;
    (cu.invP.value as Vector2).set(1 / proj.elements[0], 1 / proj.elements[5]);
    (cu.uProj.value as Matrix4).copy(proj);
    (cu.uViewInv.value as Matrix4).copy(camera.matrixWorld);
    (cu.lightDir.value as Vector3).copy(sunDirWorld).transformDirection(camera.matrixWorldInverse);
    cu.time.value = time;
    cu.theta.value = theta;
    this.quad.material = this.compositeMat;
    r.setRenderTarget(null);
    r.render(this.quadScene, this.ortho);
  }
}
