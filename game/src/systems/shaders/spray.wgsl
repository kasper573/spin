// Spray, drawn from the motes the solver flies: each a cluster of drops of one width, which
// has spread the wider the longer it has flown and is drawn out along its flight by as far as
// it flies while the eye's shutter is open. Nothing is known of the drops of a mote but how
// wide they are and how much water they hold, so nothing else is drawn of them: a mote is a
// cloud that hides as much of what is behind it as drops of that width holding that water
// cover, which is next to all of it for a mist and less and less for the same water in
// fewer, wider drops; and drops wide enough to be clear show mostly what is behind them, where
// a mist sends the light that falls on it every way.
#import bevy_pbr::mesh_view_bindings::{view, lights}
#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world}
#import optics::{diffuse_light_at, saturated, through_ring_air}
#import air::Air

struct Water {
    to_stars: vec4<f32>,
    from_water: vec4<f32>,
    origin: vec4<f32>,
    ring: vec4<f32>,
    ground: vec4<f32>,
    background: vec4<f32>,
    units: vec4<f32>,
    clock: vec4<f32>,
    absorption: vec4<f32>,
    scatter: vec4<f32>,
    air: Air,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> water: Water;
// two entries per mote: xyz where it is, w how wide its drops are; xyz its velocity, w how
// long it has flown, nought for a mote that is done
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<storage, read> motes: array<vec4<f32>>;

const PI: f32 = 3.14159265;
// how many motes a parcel is torn into: see the solver's `spray.wgsl`
const PER_PARCEL: f32 = 16.0;
// a mote is let go half as wide as a ball of its parcel's water, and its drops part at this
// share of the speed they fly at
const PARTING: f32 = 0.1;
// the widest a mote is drawn, in balls of a parcel's water: past that there is next to nothing
// of it left in any one place
const WIDEST: f32 = 6.0;
// how long the eye's shutter is open, in seconds
const SHUTTER: f32 = 1.0 / 60.0;
// drops narrower than the first of these, in metres, are a white mist, and drops wider than
// the second are clear, and show of the light falling on them only this share
const MIST: vec2<f32> = vec2(0.0005, 0.005);
const CLEAR_SHOWS: f32 = 0.25;
const WHITE: f32 = 0.9;
// drops that part from one another at random stand thickest in the middle of their mote and
// thin out from it as a bell does; a mote's width is two of the bell's deviations, and it is
// drawn out to this many of them, where next to none of it is left
const DRAWN_TO: f32 = 3.0;

struct Fragment {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    // across the mote and along it, in deviations of its bell
    @location(1) over: vec2<f32>,
    // how thickly the mote's drops stand across its middle, and how much of the light falling
    // on them they show
    @location(2) @interpolate(flat) drops: vec2<f32>,
}

@vertex
fn vertex(@location(0) numbered: vec3<f32>, @builtin(instance_index) instance: u32) -> Fragment {
    var out: Fragment;
    let i = u32(numbered.x);
    let mote = i / 6u;
    let placed = motes[2u * mote];
    let motion = motes[2u * mote + 1u];
    if (motion.w <= 0.0) {
        // a mote that is done: park the vertex outside the clip volume
        out.clip = vec4(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let world_from_local = get_world_from_local(instance);
    let centre = mesh_position_local_to_world(world_from_local, vec4(placed.xyz, 1.0)).xyz;
    let flying = (world_from_local * vec4(motion.xyz, 0.0)).xyz;
    let speed = length(motion.xyz) * water.clock.y;
    let flown = motion.w * water.units.x / water.clock.y;
    let ball = water.clock.z;
    let wide = min(0.5 * ball + PARTING * speed * flown, WIDEST * ball);
    let drawn_out = 0.5 * speed * SHUTTER;

    let to_eye = normalize(view.world_position - centre);
    var along = flying - to_eye * dot(flying, to_eye);
    if (dot(along, along) < 1e-12) {
        along = cross(to_eye, vec3(0.0, 1.0, 0.0));
    }
    along = normalize(along);
    let across = cross(to_eye, along);
    let corner = i % 6u;
    let x = select(-1.0, 1.0, corner == 1u || corner == 2u || corner == 4u);
    let y = select(-1.0, 1.0, corner == 2u || corner == 4u || corner == 5u);
    let reach = 0.5 * DRAWN_TO * vec2(wide, wide + drawn_out);
    let world = centre + across * (x * reach.x) + along * (y * reach.y);
    out.clip = view.clip_from_world * vec4(world, 1.0);
    out.world_position = world;
    out.over = vec2(x, y) * DRAWN_TO;
    // drops of one width hide three halves of their volume over that width, of whatever they
    // are spread across
    let width = placed.w * water.units.x;
    let volume = 4.0 / 3.0 * PI * ball * ball * ball / PER_PARCEL;
    let standing = 1.5 * volume / (width * 0.5 * PI * wide * (wide + drawn_out));
    let shows = mix(1.0, CLEAR_SHOWS, smoothstep(MIST.x, MIST.y, width));
    out.drops = vec2(standing, shows);
    return out;
}

@fragment
fn fragment(in: Fragment) -> @location(0) vec4<f32> {
    let out_from_middle = dot(in.over, in.over);
    let left = 1.0 - out_from_middle / (DRAWN_TO * DRAWN_TO);
    if (left <= 0.0) {
        discard;
    }
    // the bell, brought to nothing at the edge of what is drawn
    let thin = exp(-0.5 * out_from_middle) * left * left;
    let away = in.world_position - view.world_position;
    let distance = length(away);
    let covers = (1.0 - exp(-in.drops.x * thin)) * in.drops.y;
    let to_eye = -away / distance;
    // drops send the light on every way, and are as bright from the side as a white wall
    // turned half way to the light is
    var toward_light = to_eye;
    if (lights.n_directional_lights > 0u) {
        toward_light = normalize(lights.directional_lights[0].direction_to_light + to_eye);
    }
    let lit = diffuse_light_at(in.world_position, toward_light, in.clip.xy) / PI * WHITE;
    let seen = through_ring_air(lit, water.air, in.world_position + water.origin.xyz, to_eye, distance, water.ring.xy);
    return vec4(saturated(seen), covers);
}
