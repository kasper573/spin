// How light meets a smooth surface, shared by everything that reflects and lets light through:
// what a ray sees of the scene on screen, found by marching it against the depth of the scene,
// or of space beyond it; how much of the light a dielectric surface mirrors and how much it
// passes; the sun's glint off it; and the light of whatever lamps stand near a point.
#define_import_path optics

#import bevy_pbr::mesh_view_bindings::{view, lights, clustered_lights, view_transmission_texture, view_transmission_sampler}
#import bevy_pbr::mesh_view_types::POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE
#import bevy_pbr::clustered_forward::{view_fragment_cluster_index, unpack_clusterable_object_index_ranges, get_clusterable_object_id}
#import bevy_pbr::prepass_utils::prepass_depth
#import bevy_pbr::shadows::fetch_directional_shadow
#import space::stars
#import air::{Air, air_bent_each, air_crossed, air_shaft}
#import ring::{ring_run, ring_up, sun_reaches}
#import portals::{mouths, Mouths, painted, emerged, sunbeam_at, sunbeam_run, Painted, WALL, CAP}

const PI: f32 = 3.14159265;
const MARCH_STEPS: i32 = 28;
// the first step of a march, and the deepest the scene is taken to be where a ray runs past
// it, both as shares of how far off what is mirroring stands
const FINEST_MARCH: f32 = 0.01;
const THICKEST_SCENE: f32 = 0.025;
// how many times the stretch of the march that met the scene is halved to find where
const REFINE_STEPS: i32 = 5;
// the brightest the eye tells from white: the sun's mirror image is thousands of times
// brighter than that, and saturates the eye rather than flooding the view
const SATURATION: f32 = 4.0;
// how far through water anything can be seen at all
const FARTHEST: f32 = 200.0;
// the room beyond a portal, as a light: how much of the light that falls on the ring it gives
// back, which is the ground's own share; the ways out of the mouth it is looked at along, as
// the cosine and sine of their lean from straight out; and the least of a surface's sight the
// mouth may fill and still be worth working its light out for
const ROOM_ALBEDO: f32 = 0.3;
const ROOM_LEAN: vec2<f32> = vec2(0.574, 0.819);
const FAINTEST_ROOM: f32 = 1e-3;

/// A colour as the eye takes it in, saturating at the brightest it can tell apart.
fn saturated(colour: vec3<f32>) -> vec3<f32> {
    return min(colour, vec3(SATURATION));
}

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

/// A point of the world on screen: its place in the viewport and its distance in front of
/// the camera, negative when behind it.
fn screen_uv(world: vec3<f32>) -> vec3<f32> {
    let clip = view.clip_from_world * vec4(world, 1.0);
    if (clip.w <= 0.0) {
        return vec3(-1.0, -1.0, 0.0);
    }
    let ndc = clip.xy / clip.w;
    return vec3(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, clip.w);
}

fn on_screen(uv: vec2<f32>) -> bool {
    return all(uv >= vec2(0.0)) && all(uv <= vec2(1.0));
}

/// How far in front of the camera the opaque scene is at a point of the screen, whatever
/// plane the camera clips what is near it at.
fn scene_depth(uv: vec2<f32>) -> f32 {
    let coord = view.viewport.xy + uv * view.viewport.zw;
    let ndc = vec3(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, prepass_depth(vec4(coord, 0.0, 0.0), 0u));
    let seen = view.view_from_clip * vec4(ndc, 1.0);
    return -seen.z / seen.w;
}

/// The scene as drawn so far, behind everything being drawn now.
fn behind(uv: vec2<f32>) -> vec3<f32> {
    // the scene so far is kept as large as what the view is drawn into, of which the view may
    // be a window
    let kept = vec2<f32>(textureDimensions(view_transmission_texture));
    let at = (view.viewport.xy + uv * view.viewport.zw) / kept;
    return textureSampleLevel(view_transmission_texture, view_transmission_sampler, at, 0.0).rgb;
}

/// How far in front of the camera a point of the world is.
fn depth_of(world: vec3<f32>) -> f32 {
    return -(view.view_from_world * vec4(world, 1.0)).z;
}

/// Where a ray met the scene on screen, and how much of what is drawn there shows: it fades
/// out toward the edges of the screen, where the view runs out, and is nothing where the ray
/// met nothing.
struct MetOnScreen {
    uv: vec2<f32>,
    share: f32,
}

/// Where a ray from a point meets the scene on screen, found by marching it against the depth
/// of the scene no further than `reach`.
fn met_on_screen(origin: vec3<f32>, dir: vec3<f32>, reach: f32) -> MetOnScreen {
    // how far the surface itself stands from the eye sets the march: its first step and the
    // thickness it allows the scene are shares of that, so what a wall a kilometre off mirrors
    // is followed as far as what a wall a metre off does, and neither is followed finer than
    // the depth it was drawn at can tell
    let scale = max(depth_of(origin), 1e-4);
    var t = FINEST_MARCH * scale;
    var last = 0.0;
    for (var i = 0; i < MARCH_STEPS; i++) {
        let q = origin + dir * t;
        let s = screen_uv(q);
        if (t > reach || s.z <= 0.0 || !on_screen(s.xy)) {
            break;
        }
        let gap = s.z - scene_depth(s.xy);
        if (gap > 0.0) {
            // the ray can only have run past the scene by as far as the last stretch carried
            // it; anything deeper than that is behind what was drawn, not on it, and taking it
            // for a hit smears whatever is there across the reflection in streaks
            if (gap < t - last + 0.05) {
                // the scene was met somewhere over the last stretch: narrow it down
                var near = last;
                var far = t;
                for (var k = 0; k < REFINE_STEPS; k++) {
                    let mid = 0.5 * (near + far);
                    let m = screen_uv(origin + dir * mid);
                    if (m.z - scene_depth(m.xy) > 0.0) {
                        far = mid;
                    } else {
                        near = mid;
                    }
                }
                let hit = screen_uv(origin + dir * far);
                let edge = hit.xy * (1.0 - hit.xy);
                return MetOnScreen(hit.xy, smoothstep(0.0, 0.02, min(edge.x, edge.y)));
            }
            break;
        }
        last = t;
        t *= 1.28;
    }
    return MetOnScreen(vec2(0.0), 0.0);
}

/// The point of the opaque scene that is drawn at a point of the screen.
fn scene_at(uv: vec2<f32>) -> vec3<f32> {
    let coord = view.viewport.xy + uv * view.viewport.zw;
    let ndc = vec3(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, prepass_depth(vec4(coord, 0.0, 0.0), 0u));
    let world = view.world_from_clip * vec4(ndc, 1.0);
    return world.xyz / world.w;
}

/// Space in a direction of the world, given the turn that takes the world among the stars, as a
/// ray covering `spread` radians of sky sees it.
fn space_seen(dir: vec3<f32>, to_stars: vec4<f32>, background: vec3<f32>, spread: f32) -> vec3<f32> {
    return background + stars(rotate(to_stars, dir), spread);
}

/// What the air between a point of the ring and the eye does to what the eye sees of it: only
/// the stretch inside the ring holds any air, and the sun lights each stretch of it that the
/// sun reaches.
fn through_ring_air(colour: vec3<f32>, air: Air, at: vec3<f32>, to_eye: vec3<f32>, distance: f32, ring: vec2<f32>) -> vec3<f32> {
    let held = min(distance, ring_run(at, to_eye, ring).distance);
    if (held <= 0.0) {
        return colour;
    }
    // walked from the eye toward the point, which is the way the air is met
    let eye = at + to_eye * held;
    let dir = -to_eye;
    // read once rather than per sun: a uniform read inside this loop makes llvmpipe emit a
    // load from a null buffer, which brings the software renderer down
    let ambient = bounce();
    var out = colour;
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let l = lights.directional_lights[i].direction_to_light;
        let crossed = air_crossed(air, eye, dir, held, ring, sunlight(i), l, ambient);
        out = out * crossed.left + crossed.turned;
    }
    if (lights.n_directional_lights == 0u) {
        let crossed = air_crossed(air, eye, dir, held, ring, vec3(0.0), vec3(0.0, 1.0, 0.0), ambient);
        out = out * crossed.left + crossed.turned;
    }
    // a sun's light let in through a pair of portals stands in the air as a shaft
    let pair = mouths;
    if (pair.shape.z < 1.0) {
        for (var i = 0u; i < lights.n_directional_lights; i++) {
            let l = lights.directional_lights[i].direction_to_light;
            for (var k = 0u; k < 2u; k++) {
                let lit = sunbeam_run(pair, k, eye - pair.about.xyz, dir, held, l);
                if (lit.leaving > lit.entering) {
                    out += air_shaft(air, eye, dir, lit.entering, lit.leaving, ring.x, sunlight(i), lit.toward);
                }
            }
        }
    }
    return out;
}

/// Space in a direction, as it is seen from within the ring: the air the ray crosses on its way
/// out bends it, and bends each colour of it by its own amount, so a ray that crosses enough of
/// it comes out of the ring spread into its colours and what it shows is pulled out of shape
/// and fringed. Each colour is followed along its own way out.
fn space_through_air(air: Air, at: vec3<f32>, dir: vec3<f32>, ring: vec2<f32>, to_stars: vec4<f32>, background: vec3<f32>, spread: f32) -> vec3<f32> {
    let held = ring_run(at, dir, ring).distance;
    if (held <= 0.0) {
        return space_seen(dir, to_stars, background, spread);
    }
    var out = space_seen(dir, to_stars, background, spread);
    if (air.slowing.w != 0.0) {
        let ways = air_bent_each(air, at, dir, held, ring.x);
        let red = space_seen(ways[0], to_stars, background, spread);
        let green = space_seen(ways[1], to_stars, background, spread);
        let blue = space_seen(ways[2], to_stars, background, spread);
        out = vec3(red.r, green.g, blue.b);
    }
    // and the air it crossed on the way out dims it and glows in front of it
    return through_ring_air(out, air, at + dir * held, -dir, held, ring);
}

/// What a stretch of water `distance` long does to the light that set out across it: what is
/// left of that light, and the water's own colour gathered along the way.
///
/// Absorption and scattering both take light out of a ray and the share scattering takes is
/// the share that comes back, so water deep enough to hide whatever lies beyond it settles at
/// that share of the light falling on it. The light falling on it comes down from the surface,
/// so a stretch lit that way shows its colour the brighter the shallower it lies: `rise` is
/// how fast the ray climbs toward the surface as it runs away from the eye, one straight up
/// and minus one straight down, and `glow` is the colour the water shows where the ray sets
/// out. Looking straight up the water brightens exactly as fast as the ray dims, so the two
/// cancel and the colour gathers evenly the whole way; looking down it falls away twice as
/// fast as it would level.
fn through_water(colour: vec3<f32>, glow: vec3<f32>, extinction: vec3<f32>, rise: f32, distance: f32) -> vec3<f32> {
    let d = min(distance, FARTHEST);
    let climb = max(1.0 - rise, 1e-3);
    return colour * exp(-extinction * d) + glow * (1.0 - exp(-extinction * climb * d)) / climb;
}

/// The suns' light falling on a matte surface of the ring at a point of its frame, facing `n`,
/// where each reaches past the ring.
fn ring_sunlight(hit: vec3<f32>, n: vec3<f32>, ring: vec2<f32>) -> vec3<f32> {
    var light = vec3(0.0);
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let l = lights.directional_lights[i].direction_to_light;
        let ndl = dot(n, l);
        if (ndl > 0.0 && sun_reaches(hit, l, ring)) {
            light += sunlight(i) * ndl / PI;
        }
    }
    return light;
}

/// All the light falling there: the bounce light everywhere, and the suns'.
fn ring_light(hit: vec3<f32>, n: vec3<f32>, ring: vec2<f32>) -> vec3<f32> {
    return bounce() + ring_sunlight(hit, n, ring);
}

/// What a ray meets once it has left the screen, inside the ring: the ground where it strikes
/// the ring, lit by the sun where the sun reaches it and by the bounce light everywhere, or
/// space where it leaves through a cap; and whatever portal is let into the wall or the cap
/// where it gets there, which the ray goes on through where that is open.
fn ring_seen(air: Air, at: vec3<f32>, dir: vec3<f32>, ring: vec2<f32>, ground: vec3<f32>, to_stars: vec4<f32>, background: vec3<f32>, spread: f32) -> vec3<f32> {
    let run = ring_run(at, dir, ring);
    let hit = at + dir * run.distance;
    var on = CAP;
    var n = vec3(0.0, -sign(dir.y), 0.0);
    if (run.wall) {
        on = WALL;
        n = ring_up(hit, ring.x);
    }
    let portal = painted(hit - mouths.about.xyz, on, true, max(spread * run.distance, 1e-3));
    var beyond = vec3(0.0);
    if (portal.paint + portal.open < 1.0) {
        beyond = ring_met(air, at, dir, ring, ground, to_stars, background, spread);
    }
    if (portal.paint + portal.open <= 0.0 && all(portal.glow == vec3(0.0))) {
        return beyond;
    }
    var shown = portal.albedo * ring_light(hit, n, ring) * portal.paint + portal.glow;
    if (portal.open > 0.0) {
        let far = emerged(portal.mouth, hit - mouths.about.xyz, dir);
        shown += ring_met(air, far.origin + mouths.about.xyz, far.dir, ring, ground, to_stars, background, spread) * portal.open;
    }
    // the air between here and the portal dims what the portal shows and glows before it as
    // it does before the wall beside it, whose share of that glow is already in what was met
    let glow_of_air = through_ring_air(vec3(0.0), air, hit, -dir, run.distance, ring);
    let through_air = through_ring_air(shown, air, hit, -dir, run.distance, ring);
    let covered = portal.paint + portal.open;
    return beyond * (1.0 - covered) + through_air - glow_of_air * (1.0 - covered);
}

/// The ring as a ray meets it with no portals in it: the ground, or space through a cap.
fn ring_met(air: Air, at: vec3<f32>, dir: vec3<f32>, ring: vec2<f32>, ground: vec3<f32>, to_stars: vec4<f32>, background: vec3<f32>, spread: f32) -> vec3<f32> {
    let run = ring_run(at, dir, ring);
    if (!run.wall) {
        return space_through_air(air, at, dir, ring, to_stars, background, spread);
    }
    let hit = at + dir * run.distance;
    // the wall is seen through whatever air stands between it and where the ray set out
    return through_ring_air(ground * ring_light(hit, ring_up(hit, ring.x), ring), air, hit, -dir, run.distance, ring);
}

/// What is seen through a surface at a point of the screen, refracted to `exit`: the scene
/// there unless something stands in front of the surface, or the refraction leaves the screen.
fn seen_through(uv: vec2<f32>, depth_here: f32, exit: vec3<f32>) -> vec3<f32> {
    var s = screen_uv(exit);
    if (s.z <= 0.0 || !on_screen(s.xy) || nearest_scene_depth(s.xy) < depth_here) {
        s = vec3(uv, depth_here);
    }
    return behind(s.xy);
}

/// The nearest the opaque scene comes to the camera round a point of the screen. What is
/// drawn there is blended from the pixels round it, so something standing in front of any of
/// them shows in it.
fn nearest_scene_depth(uv: vec2<f32>) -> f32 {
    let pixel = 1.0 / view.viewport.zw;
    var nearest = scene_depth(uv);
    nearest = min(nearest, scene_depth(uv + vec2(pixel.x, pixel.y)));
    nearest = min(nearest, scene_depth(uv + vec2(-pixel.x, pixel.y)));
    nearest = min(nearest, scene_depth(uv + vec2(pixel.x, -pixel.y)));
    nearest = min(nearest, scene_depth(uv + vec2(-pixel.x, -pixel.y)));
    return nearest;
}

/// Schlick's share of light a dielectric mirrors at this angle, given what it mirrors face on.
fn fresnel(cos_theta: f32, f0: f32) -> f32 {
    let m = 1.0 - clamp(cos_theta, 0.0, 1.0);
    return f0 + (1.0 - f0) * m * m * m * m * m;
}

/// The sun's glint: a GGX lobe, tight on a still surface and spread by its roughness.
fn glint(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32, f0: f32) -> f32 {
    let nl = dot(n, l);
    let nv = dot(n, v);
    if (nl <= 0.0 || nv <= 0.0) {
        return 0.0;
    }
    let h = normalize(l + v);
    let nh = max(dot(n, h), 0.0);
    let a = roughness * roughness;
    let a2 = a * a;
    let d = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    let vis = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2) + 1e-4);
    return d * vis * fresnel(max(dot(v, h), 0.0), f0) * nl;
}

/// The light bounced round the ring, as the eye is exposed to it.
fn bounce() -> vec3<f32> {
    return lights.ambient_color.rgb * view.exposure;
}

/// A sun's light, as the eye is exposed to it.
fn sunlight(i: u32) -> vec3<f32> {
    return lights.directional_lights[i].color.rgb * view.exposure;
}

/// How much of a sun's light reaches a point of the world, given the ring's shadow.
fn sun_shadow(i: u32, world: vec3<f32>, n: vec3<f32>, pixel: vec2<f32>) -> f32 {
    let view_z = -(view.view_from_world * vec4(world, 1.0)).z;
    return fetch_directional_shadow(i, vec4(world, 1.0), n, view_z, pixel);
}

/// The lamps whose light reaches a point of the world drawn at a pixel, as a stretch of the
/// list of them: its first, the first of those that throw their light in a cone only, and one
/// past its last.
fn lamps_near(world: vec3<f32>, pixel: vec2<f32>) -> vec3<u32> {
    let view_z = (view.view_from_world * vec4(world, 1.0)).z;
    let near = unpack_clusterable_object_index_ranges(view_fragment_cluster_index(pixel, view_z, false));
    return vec3(near.first_point_light_index_offset, near.first_spot_light_index_offset, near.first_reflection_probe_index_offset);
}

/// One lamp of that list as a point of the world has it: which way its light comes from, and
/// that light, falling off with the square of how far it has come, before the eye's exposure.
struct Lamp {
    toward: vec3<f32>,
    light: vec3<f32>,
}

fn lamp_at(listed: u32, lamps: vec3<u32>, world: vec3<f32>) -> Lamp {
    let lamp = clustered_lights.data[get_clusterable_object_id(listed)];
    let reach = lamp.position_radius.xyz - world;
    let far2 = dot(reach, reach);
    let toward = reach * inverseSqrt(max(far2, 1e-8));
    // eased to nothing at the lamp's range, as the lit materials around it have it
    let ranged = far2 * lamp.color_inverse_square_range.w;
    let eased = saturate(1.0 - ranged * ranged);
    var light = lamp.color_inverse_square_range.rgb * (eased * eased / max(far2, 1e-4));
    if (listed >= lamps.y) {
        // a cone of light, as the lit materials around it have it
        var axis = vec3(lamp.light_custom_data.x, 0.0, lamp.light_custom_data.y);
        axis.y = sqrt(max(0.0, 1.0 - axis.x * axis.x - axis.z * axis.z));
        if ((lamp.flags & POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE) != 0u) {
            axis.y = -axis.y;
        }
        let within = saturate(-dot(axis, toward) * lamp.light_custom_data.z + lamp.light_custom_data.w);
        light *= within * within;
    }
    return Lamp(toward, light);
}

/// The light the lamps near a point throw on a matte surface there, facing `n`, as the eye is
/// exposed to it.
fn lamplight_at(world: vec3<f32>, n: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    let lamps = lamps_near(world, pixel);
    var light = vec3(0.0);
    for (var i = lamps.x; i < lamps.z; i++) {
        let lamp = lamp_at(i, lamps, world);
        light += lamp.light * max(dot(n, lamp.toward), 0.0);
    }
    return light * view.exposure;
}

/// The lamps' glint off a smooth surface at a point, as the eye is exposed to it.
fn lamp_glint_at(world: vec3<f32>, n: vec3<f32>, v: vec3<f32>, roughness: f32, f0: f32, pixel: vec2<f32>) -> vec3<f32> {
    let lamps = lamps_near(world, pixel);
    var light = vec3(0.0);
    for (var i = lamps.x; i < lamps.z; i++) {
        let lamp = lamp_at(i, lamps, world);
        light += lamp.light * glint(n, v, lamp.toward, roughness, f0);
    }
    return light * view.exposure;
}

/// The suns' light that reaches a matte surface at a point of the world through a pair of
/// portals, facing `n`, as the eye is exposed to it.
fn sunbeams_at(world: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let pair = mouths;
    var light = vec3(0.0);
    if (pair.shape.z >= 1.0) {
        return light;
    }
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let l = lights.directional_lights[i].direction_to_light;
        for (var k = 0u; k < 2u; k++) {
            let beam = sunbeam_at(pair, k, world, l);
            light += sunlight(i) * (beam.w * max(dot(n, beam.xyz), 0.0));
        }
    }
    return light;
}

/// The glint of that light off a smooth surface there.
fn sunbeam_glints_at(world: vec3<f32>, n: vec3<f32>, v: vec3<f32>, roughness: f32, f0: f32) -> vec3<f32> {
    let pair = mouths;
    var light = vec3(0.0);
    if (pair.shape.z >= 1.0) {
        return light;
    }
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let l = lights.directional_lights[i].direction_to_light;
        for (var k = 0u; k < 2u; k++) {
            let beam = sunbeam_at(pair, k, world, l);
            if (beam.w > 0.0) {
                light += sunlight(i) * (beam.w * glint(n, v, beam.xyz, roughness, f0));
            }
        }
    }
    return light;
}

/// The light of the room beyond mouth `k` of a pair that reaches a matte surface at a point
/// before it, facing `n`: the mouth is a disc as bright as what the other mouth looks out on,
/// which is the ring across from it, met a few ways out of it and lit as the ring is.
fn room_beyond(pair: Mouths, k: u32, world: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let near = pair.mouths[k];
    let far = pair.mouths[1u - k];
    let opening = pair.shape.y * (1.0 - pair.shape.z);
    let reach = near.centre.xyz - world;
    let far2 = dot(reach, reach);
    let toward = reach * inverseSqrt(max(far2, 1e-8));
    // how much of what the surface sees the disc fills, as each faces the other
    let filled = opening * opening * max(dot(n, toward), 0.0) * max(dot(near.inward.xyz, -toward), 0.0) / (far2 + opening * opening);
    if (near.origin.w == 0.0 || far.origin.w == 0.0 || filled < FAINTEST_ROOM) {
        return vec3(0.0);
    }
    let ring = vec2(pair.shape.x, pair.look.z);
    let looked_from = far.centre.xyz + pair.about.xyz;
    let ambient = bounce();
    var seen = vec3(0.0);
    for (var way = 0u; way < 4u; way++) {
        let side = select(far.spinward.xyz, far.upward.xyz, way >= 2u) * select(-1.0, 1.0, (way & 1u) == 1u);
        let dir = normalize(far.inward.xyz * ROOM_LEAN.x + side * ROOM_LEAN.y);
        let run = ring_run(looked_from + far.inward.xyz * 1e-3 * opening, dir, ring);
        if (run.wall) {
            let met = looked_from + dir * run.distance;
            seen += (ambient + ring_sunlight(met, ring_up(met, ring.x), ring)) * (0.25 * ROOM_ALBEDO);
        }
    }
    return seen * (PI * filled);
}

/// The light that reaches a matte surface at a point of the world from the rooms beyond a
/// pair of portals, as the eye is exposed to it.
fn rooms_beyond_at(world: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let pair = mouths;
    if (pair.shape.z >= 1.0) {
        return vec3(0.0);
    }
    return room_beyond(pair, 0u, world, n) + room_beyond(pair, 1u, world, n);
}

/// The light falling on a matte surface at a point of the world, facing `n`, where the ring
/// shades it from the suns.
fn diffuse_light_at(world: vec3<f32>, n: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    var light = bounce() + lamplight_at(world, n, pixel) + sunbeams_at(world, n) + rooms_beyond_at(world, n);
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let l = lights.directional_lights[i].direction_to_light;
        let ndl = dot(n, l);
        if (ndl > 0.0) {
            light += sunlight(i) * ndl * sun_shadow(i, world, n, pixel);
        }
    }
    return light;
}
