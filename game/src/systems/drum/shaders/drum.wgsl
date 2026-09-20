// The drum as the fluid solver's vessel: a spinning cylinder with caps and a heightfield
// landscape on its inside wall, in its own turning frame and the water's units. The water's
// frame sits on the wall at the water's own site: x out through the glass, y along the axis and
// z spinward, with the axis `radius` in along -x. Everything is worked out from those small
// coordinates without forming the radius squared or any sum it would swamp, so it holds on a
// ring of any size. Mirrors `Drum` on the CPU.
#define_import_path vessel
#import ring::short_way_round

struct DrumUniform {
    // the drum's spin and its rate of change, per second of the water's clock
    spin: f32,
    spin_rate: f32,
    // the drum's size in the water's units, how many of them a metre is and how many of the
    // water's seconds a second, and where the water's site is along the axis from the middle
    radius: f32,
    half_width: f32,
    per_metre: f32,
    per_second: f32,
    along: f32,
    // the depth the ground was laid with, and how many of its cells a unit of the water spans
    // round the ring and along it
    base: f32,
    cells_round: f32,
    cells_along: f32,
    // how far across its cell the water's site lies, which cell of its patch that is, and the
    // rows either side of the middle one
    across: f32,
    first: i32,
    rows: i32,
    // how many patches there are round the ring, or 0 when there are too many for the water
    // to reach round it; how many are sculpted; the mask of the table they are found by; and
    // how many of them a row of the atlas holds
    patches_round: u32,
    patches: u32,
    table_mask: u32,
    atlas: u32,
    // how far from the middle of a mouth it is open, which is nowhere while it is shut
    passable: f32,
    openings: array<Opening, 2>,
}

/// A mouth of the pair the drum may be joined to itself by: its middle, with 1 when it stands;
/// the way across it and up it on its surface; and the way off its surface into the room, with
/// 1 when it is on the wall, where how far off the surface something is is its height over the
/// ground, rather than on a cap, which is flat.
struct Opening {
    centre: vec4<f32>,
    across: vec4<f32>,
    up: vec4<f32>,
    inward: vec4<f32>,
}

@group(1) @binding(0) var<uniform> drum: DrumUniform;
// every sculpted patch's heights in metres, a square of the atlas each, along the axis across
// and round the ring down
@group(1) @binding(1) var ground: texture_2d<f32>;
// the table the patches are found by: which patch round the ring from the water's site's, which
// along the axis, and one more than where its square is in the atlas, or 0 for an empty entry
@group(1) @binding(2) var patches: texture_2d<i32>;

const PATCH: i32 = 32;
const TABLE_WIDTH: u32 = 256u;

struct Confined {
    p: vec3<f32>,
    // inward wall normals, w = 1 when present
    first: vec4<f32>,
    second: vec4<f32>,
}

/// Where free flight lands, and at what velocity in the frame.
struct Flight {
    p: vec3<f32>,
    v: vec3<f32>,
}

struct Penetration {
    depth: f32,
    normal: vec3<f32>,
}

/// How far a point is inside the glass, and the way out through it there.
struct Wall {
    height: f32,
    outward: vec3<f32>,
}

fn wall(p: vec3<f32>) -> Wall {
    let a = drum.radius + p.x;
    let r = sqrt(a * a + p.z * p.z);
    var out: Wall;
    out.outward = vec3(1.0, 0.0, 0.0);
    if (r > 0.0) {
        out.outward = vec3(a / r, 0.0, p.z / r);
    }
    out.height = -(2.0 * drum.radius * p.x + p.x * p.x + p.z * p.z) / (r + drum.radius);
    return out;
}

/// Where a sculpted patch's square is in the atlas, or -1 if the patch is as it was laid.
fn patch_slot(round: i32, along: i32) -> i32 {
    var h = (u32(round) * 0x9e3779b1u) ^ (u32(along) * 0x85ebca77u);
    h ^= h >> 15u;
    var slot = h & drum.table_mask;
    for (var probe = 0u; probe <= drum.table_mask; probe++) {
        let entry = textureLoad(patches, vec2<i32>(i32(slot % TABLE_WIDTH), i32(slot / TABLE_WIDTH)), 0);
        if (entry.z == 0) {
            return -1;
        }
        if (entry.x == round && entry.y == along) {
            return entry.z - 1;
        }
        slot = (slot + 1u) & drum.table_mask;
    }
    return -1;
}

/// The patch a cell lies in: counted in cells from the start of the water's site's patch round
/// the ring, and from the middle along it.
fn patch_of(cell: i32, row: i32) -> i32 {
    var round = cell >> 5u;
    if (drum.patches_round > 0u) {
        round = short_way_round(round, i32(drum.patches_round));
    }
    return patch_slot(round, row >> 5u);
}

fn height_in(slot: i32, cell: i32, row: i32) -> f32 {
    if (slot < 0) {
        return drum.base;
    }
    let s = u32(slot);
    let texel = vec2<i32>(
        i32((s % drum.atlas) * u32(PATCH)) + (row & 31),
        i32((s / drum.atlas) * u32(PATCH)) + (cell & 31),
    );
    return textureLoad(ground, texel, 0).r * drum.per_metre;
}

/// The heights at a cell and the three past it round the ring and along it, looked up once
/// when all four are in the one patch.
fn corners(cell: i32, row: i32) -> vec4<f32> {
    if ((cell & 31) < 31 && (row & 31) < 31) {
        let slot = patch_of(cell, row);
        return vec4(
            height_in(slot, cell, row),
            height_in(slot, cell + 1, row),
            height_in(slot, cell, row + 1),
            height_in(slot, cell + 1, row + 1),
        );
    }
    return vec4(
        height_in(patch_of(cell, row), cell, row),
        height_in(patch_of(cell + 1, row), cell + 1, row),
        height_in(patch_of(cell, row + 1), cell, row + 1),
        height_in(patch_of(cell + 1, row + 1), cell + 1, row + 1),
    );
}

/// The ground under a point: how high it stands over the glass, and how fast it rises round the
/// ring and along the axis, per unit.
fn ground_under(p: vec3<f32>) -> vec3<f32> {
    if (drum.patches == 0u) {
        return vec3(drum.base, 0.0, 0.0);
    }
    let arc = drum.radius * atan2(p.z, drum.radius + p.x);
    let u = drum.across + arc * drum.cells_round;
    let rows = f32(drum.rows);
    let v = clamp((drum.along + p.y) * drum.cells_along, -rows, rows - 1e-3);
    let i = floor(u);
    let j = floor(v);
    let fu = u - i;
    let fv = v - j;
    let h = corners(drum.first + i32(i), i32(j));
    let height = mix(mix(h.x, h.y, fu), mix(h.z, h.w, fu), fv);
    let round = ((h.y - h.x) * (1.0 - fv) + (h.w - h.z) * fv) * drum.cells_round;
    let along = ((h.z - h.x) * (1.0 - fu) + (h.w - h.y) * fu) * drum.cells_along;
    return vec3(height, round, along);
}

/// Signed penetration of a point into the terrain (positive = inside) and the inward normal,
/// with the terrain's surface brought inward by `margin` square to itself: how far the ground
/// stands over the point says how far into it the point is only by how steep the ground is.
fn landscape_penetration(p: vec3<f32>, margin: f32) -> Penetration {
    let a = drum.radius + p.x;
    let r2 = a * a + p.z * p.z;
    if (r2 < 1e-12) {
        return Penetration(-drum.radius, vec3(0.0));
    }
    let w = wall(p);
    let g = ground_under(p);
    let f = g.x - w.height;
    // the ground's rise round the ring is per unit of arc at the glass, and the point's angle
    // round the axis changes by its move across the radial line over its distance from the axis
    let k = g.y * drum.radius / r2;
    let grad = vec3(w.outward.x - k * p.z, g.z, w.outward.z + k * a);
    let len = max(length(grad), 1e-12);
    return Penetration(f / len + margin, -grad / len);
}

const VESSEL_OPENINGS: u32 = 2u;

/// A point as it is beyond an opening, where it is near enough to it to be, and where it is
/// in the chart of the opening it is before.
struct Through {
    p: vec3<f32>,
    there: bool,
    opening: u32,
    chart: vec3<f32>,
}

fn pair_is_open() -> bool {
    return drum.passable > 0.0 && drum.openings[0].centre.w > 0.0 && drum.openings[1].centre.w > 0.0;
}

fn height_over_ground(p: vec3<f32>) -> f32 {
    return wall(p).height - ground_under(p).x;
}

/// A point in a mouth's chart: across it, up it, and off its surface into the room.
fn charted(o: Opening, p: vec3<f32>) -> vec3<f32> {
    let d = p - o.centre.xyz;
    var h = dot(d, o.inward.xyz);
    if (o.inward.w > 0.0) {
        h = height_over_ground(p);
    }
    return vec3(dot(d, o.across.xyz), dot(d, o.up.xyz), h);
}

/// The point of a mouth's chart that the other mouth's `c` is joined to: as far across it the
/// other way, as far up it, and as far off its surface on the other side of it.
fn joined_to(o: Opening, c: vec3<f32>) -> vec3<f32> {
    let flat = o.centre.xyz - o.across.xyz * c.x + o.up.xyz * c.y;
    if (o.inward.w > 0.0) {
        let w = wall(flat);
        return flat + w.outward * (w.height - ground_under(flat).x + c.z);
    }
    return flat - o.inward.xyz * c.z;
}

/// How far from a mouth's middle a point is, along its surface as nearly as a flat one tells.
fn across_opening(o: Opening, p: vec3<f32>) -> f32 {
    let d = p - o.centre.xyz;
    let off = dot(d, o.inward.xyz);
    return sqrt(max(dot(d, d) - off * off, 0.0));
}

/// The mouth that the ground or the cap a point is at is open by, or -1.
fn opening_at(p: vec3<f32>, on_the_wall: f32) -> i32 {
    if (!pair_is_open()) {
        return -1;
    }
    for (var i = 0u; i < VESSEL_OPENINGS; i++) {
        let o = drum.openings[i];
        let d = p - o.centre.xyz;
        if (o.inward.w == on_the_wall && dot(d, d) < 4.0 * drum.passable * drum.passable
            && across_opening(o, p) < drum.passable) {
            return i32(i);
        }
    }
    return -1;
}

/// What keeps a point over what is open of a mouth `margin` clear of the mouth's edge, which
/// is all there is of the surface there: how far it is pushed, and which way.
fn lip(o: Opening, p: vec3<f32>, margin: f32) -> Penetration {
    let d = p - o.centre.xyz;
    let flat = d - o.inward.xyz * dot(d, o.inward.xyz);
    let across = length(flat);
    let gap = drum.passable - across;
    let h = charted(o, p).z;
    let clear = sqrt(gap * gap + h * h);
    if (clear >= margin || across < 1e-6) {
        return Penetration(0.0, o.inward.xyz);
    }
    var away = o.inward.xyz;
    if (clear > 1e-6) {
        away = (o.inward.xyz * h - flat / across * gap) / clear;
    }
    return Penetration(margin - clear, away);
}

/// Where a point within `reach` of what is open of a mouth is beyond it.
fn vessel_through(p: vec3<f32>, opening: u32, reach: f32) -> Through {
    var out = Through(p, false, opening, vec3(0.0));
    if (!pair_is_open()) {
        return out;
    }
    let o = drum.openings[opening];
    let d = p - o.centre.xyz;
    let span = drum.passable + reach;
    if (dot(d, d) >= span * span) {
        return out;
    }
    out.chart = charted(o, p);
    out.p = joined_to(drum.openings[1u - opening], out.chart);
    out.there = true;
    return out;
}

/// Where a point that has sunk into what is open of a mouth has come out of the other.
fn vessel_gone_through(p: vec3<f32>) -> Through {
    var out = Through(p, false, 0u, vec3(0.0));
    if (!pair_is_open()) {
        return out;
    }
    for (var i = 0u; i < VESSEL_OPENINGS; i++) {
        let o = drum.openings[i];
        let d = p - o.centre.xyz;
        if (dot(d, d) >= 4.0 * drum.passable * drum.passable || across_opening(o, p) >= drum.passable) {
            continue;
        }
        let c = charted(o, p);
        if (c.z < 0.0) {
            return Through(joined_to(drum.openings[1u - i], c), true, i, c);
        }
    }
    return out;
}

/// A vector carried through a mouth: turned as the two surfaces lie.
fn vessel_turned(v: vec3<f32>, opening: u32) -> vec3<f32> {
    let o = drum.openings[opening];
    let far = drum.openings[1u - opening];
    return -dot(v, o.across.xyz) * far.across.xyz + dot(v, o.up.xyz) * far.up.xyz
        - dot(v, o.inward.xyz) * far.inward.xyz;
}

/// From a point beyond a mouth to a point before it that was brought through, as the side
/// the point was brought from has it, with 1 where the straight way between them leads through
/// what is open of the mouth. The two are told apart in the chart the mouths share, where the
/// one is as far off the mouths' surface on its side as the other is on the other's: so that
/// each of the two has the other exactly as far off, and the opposite way, which what they
/// do to each other rests on, however the ground the mouths are let into is shaped.
fn vessel_apart(brought: Through, other: vec3<f32>) -> vec4<f32> {
    let o = drum.openings[brought.opening];
    let c = brought.chart;
    let d = charted(drum.openings[1u - brought.opening], other);
    let r = vec3(c.x + d.x, c.y - d.y, c.z + d.z);
    var t = 0.5;
    if (c.z + d.z > 0.0) {
        t = clamp(c.z / (c.z + d.z), 0.0, 1.0);
    }
    let crossing = mix(c.xy, vec2(-d.x, d.y), t);
    if (dot(crossing, crossing) >= drum.passable * drum.passable) {
        return vec4(0.0);
    }
    return vec4(o.across.xyz * r.x + o.up.xyz * r.y + o.inward.xyz * r.z, 1.0);
}

fn vessel_confine(p_in: vec3<f32>, margin: f32) -> Confined {
    var p = p_in;
    var out = Confined(p, vec4(0.0), vec4(0.0));
    var count = 0u;
    let hole = opening_at(p, 1.0);
    if (hole >= 0) {
        // no ground stands where a mouth is open, only the mouth's edge
        let pen = lip(drum.openings[hole], p, margin);
        if (pen.depth > 0.0) {
            p += pen.normal * pen.depth;
            out.first = vec4(pen.normal, 1.0);
            count = 1u;
        }
    } else if (drum.patches == 0u) {
        // the ground is level all round, so it is the glass brought in by its depth
        let w = wall(p);
        let lift = drum.base + margin - w.height;
        if (lift > 0.0) {
            p -= w.outward * lift;
            out.first = vec4(-w.outward, 1.0);
            count = 1u;
        }
    } else {
        for (var attempt = 0u; attempt < 2u; attempt++) {
            let pen = landscape_penetration(p, margin);
            if (pen.depth <= 0.0) {
                break;
            }
            p += pen.normal * pen.depth;
            if (count == 0u) {
                out.first = vec4(pen.normal, 1.0);
                count = 1u;
            }
        }
    }
    let cap = drum.half_width - margin;
    let axial = drum.along + p.y;
    var cap_normal = vec4(0.0);
    let port = opening_at(p, 0.0);
    if (port >= 0) {
        let pen = lip(drum.openings[port], p, margin);
        if (pen.depth > 0.0) {
            p += pen.normal * pen.depth;
            cap_normal = vec4(pen.normal, 1.0);
        }
    } else if (axial > cap) {
        p.y = cap - drum.along;
        cap_normal = vec4(0.0, -1.0, 0.0, 1.0);
    } else if (axial < -cap) {
        p.y = -cap - drum.along;
        cap_normal = vec4(0.0, 1.0, 0.0, 1.0);
    }
    if (cap_normal.w > 0.0) {
        if (count == 0u) {
            out.first = cap_normal;
        } else {
            out.second = cap_normal;
        }
    }
    out.p = p;
    return out;
}

/// What a point at rest in the turning drum is accelerated by: flung outward by the spin, and
/// left behind as the spin changes.
fn vessel_gravity(p: vec3<f32>) -> vec3<f32> {
    let w = drum.spin;
    let a = drum.spin_rate;
    let out = drum.radius + p.x;
    return vec3(w * w * out - a * p.z, 0.0, w * w * p.z + a * out);
}

/// A point turned about the axis through the frame's origin by the turn whose sine and cosine
/// these are.
fn turned(p: vec3<f32>, s: f32, c: f32) -> vec3<f32> {
    return vec3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

/// sin(t) and 1 - cos(t). The builtins are held only to an absolute error, which is all there is
/// of the turn a big ring's frame makes in a step, so a small turn is worked out as its series.
fn sines(t: f32) -> vec2<f32> {
    if (abs(t) < 0.1) {
        let t2 = t * t;
        return vec2(
            t * (1.0 - t2 / 6.0 * (1.0 - t2 / 20.0 * (1.0 - t2 / 42.0))),
            0.5 * t2 * (1.0 - t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0))),
        );
    }
    let half = sin(0.5 * t);
    return vec2(sin(t), 2.0 * half * half);
}

/// sin(t) - t cos(t), which is t³/3 for small t and would be lost to rounding worked out as
/// the difference it is.
fn sine_lag(t: f32) -> f32 {
    if (abs(t) < 0.1) {
        let t2 = t * t;
        return t * t2 * (1.0 / 3.0 - t2 * (1.0 / 30.0 - t2 / 840.0));
    }
    return sin(t) - t * cos(t);
}

/// Free flight through the drum's frame over `dt`, which ends at the spin the frame is at:
/// straight and even among the stars, put back into the frame where it has turned to by the
/// end, so that free-flying water keeps its motion among the stars exactly. The frame's origin
/// is carried round the axis at the speed of the rim, which the motion is taken apart from:
/// where that speed and the frame's turn nearly cancel, what is left is worked out as what it
/// comes to rather than as the difference of the two.
fn vessel_flight(p: vec3<f32>, v: vec3<f32>, dt: f32) -> Flight {
    let before = drum.spin - drum.spin_rate * dt;
    let turn = drum.spin * dt - 0.5 * drum.spin_rate * dt * dt;
    // how far the frame turned beyond what its spin at the start would have turned it
    let ahead = 0.5 * drum.spin_rate * dt * dt;
    let sine = sines(turn);
    let s = sine.x;
    let sagged = sine.y;
    let c = 1.0 - sagged;
    let among_stars = v + vec3(before * p.z, 0.0, -before * p.x);
    // where the frame's origin went among the stars, seen from the frame it turned into
    let origin = drum.radius
        * vec3(before * dt * s - sagged, 0.0, sine_lag(turn) + ahead * c);
    let landed = turned(p + among_stars * dt, -s, c) + origin;
    let carried = turned(among_stars, -s, c)
        + drum.radius * vec3(before * s, 0.0, drum.spin_rate * dt + before * sagged);
    var out: Flight;
    out.p = landed;
    out.v = carried - vec3(drum.spin * landed.z, 0.0, -drum.spin * landed.x);
    return out;
}

/// The velocity, in the frame, of something at rest among the stars at `p`.
fn vessel_star_velocity(p: vec3<f32>) -> vec3<f32> {
    return vec3(-drum.spin * p.z, 0.0, drum.spin * (drum.radius + p.x));
}

fn vessel_has_air(p: vec3<f32>) -> bool {
    // inside the glass, where the point is nearer the axis than the glass is
    let inside = 2.0 * drum.radius * p.x + p.x * p.x + p.z * p.z < 0.0;
    return inside && abs(drum.along + p.y) < drum.half_width;
}
