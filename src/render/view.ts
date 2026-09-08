import {
  Color,
  DepthTexture,
  DirectionalLight,
  HemisphereLight,
  Scene,
  UnsignedIntType,
  Vector3,
  WebGLRenderer,
  WebGLRenderTarget,
  HalfFloatType,
} from 'three';
import type { SimState } from '../physics/world';
import { FlyCamera } from './camera';
import { Markers, type MarkerKind } from './markers';
import { RaftMeshes } from './rafts';
import { makeStars } from './stars';
import { WaterPass } from './water';
import { Wheel } from './wheel';

const SPACE = 0x05070d;
const SUN_DIR = new Vector3(6, 9, 4).normalize();
const FILL_DIR = new Vector3(-5, -6, -3).normalize();

/**
 * Frame composition:
 *   1. opaque scene (stars, cage, rafts) → colour + depth target
 *   2. water pass → screen, refracting the scene target and writing depth
 *   3. glass shell and cursor markers blended on top
 */
export class View {
  readonly renderer: WebGLRenderer;
  readonly fly = new FlyCamera();
  private readonly scene = new Scene();
  private readonly glassScene = new Scene();
  private readonly overlayScene = new Scene();
  private readonly sceneTarget: WebGLRenderTarget;
  private readonly water: WaterPass;
  private readonly wheel = new Wheel();
  private readonly rafts: RaftMeshes;
  private readonly markers: Markers;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.autoClear = false;

    this.scene.background = new Color(SPACE);
    this.scene.add(new HemisphereLight(0xcfe0ff, 0x0b1020, 0.75));
    const sun = new DirectionalLight(0xfff4e0, 1.1);
    sun.position.copy(SUN_DIR).multiplyScalar(10);
    const fill = new DirectionalLight(0x6a8cff, 0.35);
    fill.position.copy(FILL_DIR).multiplyScalar(10);
    this.scene.add(sun, fill, makeStars(), this.wheel.frame);
    this.glassScene.add(this.wheel.glass);
    this.rafts = new RaftMeshes(this.scene);
    this.markers = new Markers(this.overlayScene);

    this.sceneTarget = new WebGLRenderTarget(1, 1, {
      type: HalfFloatType,
      depthTexture: new DepthTexture(1, 1, UnsignedIntType),
      depthBuffer: true,
      stencilBuffer: false,
      samples: 4,
    });
    this.water = new WaterPass(this.renderer);
    this.resize();
  }

  resize(): void {
    const w = window.innerWidth,
      h = window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.fly.setAspect(w / h);
    const pw = Math.round(w * this.renderer.getPixelRatio()),
      ph = Math.round(h * this.renderer.getPixelRatio());
    this.sceneTarget.setSize(pw, ph);
    this.water.resize(pw, ph);
  }

  render(
    S: SimState,
    marker: MarkerKind,
    aimPoint: Vector3,
    aimNormal: Vector3,
    raftOffset: number,
    brushRadius: number,
  ): void {
    const r = this.renderer,
      cam = this.fly.camera;
    this.wheel.update(S.theta, cam, SUN_DIR, FILL_DIR);
    this.wheel.landscape.sync(S.landscape);
    this.rafts.sync(S.rafts);
    this.markers.update(marker, aimPoint, aimNormal, raftOffset, brushRadius);
    this.water.upload(S.fluid);

    r.setRenderTarget(this.sceneTarget);
    r.setClearColor(SPACE, 1);
    r.clear(true, true, false);
    r.render(this.scene, cam);

    this.water.render(cam, this.sceneTarget, SUN_DIR, S.time, -S.theta);

    r.setRenderTarget(null);
    r.render(this.glassScene, cam);
    r.render(this.overlayScene, cam);
  }
}
