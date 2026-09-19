// Where a particle's neighbours are. A vessel may be joined to itself through openings in its
// walls, and what is just before an opening is next to what is just beyond it: so a particle
// near an opening has neighbours in more than one place, the water round it and the water
// round where it is beyond the opening. Each such place is a view: the particle's own first,
// then one through each opening it is within reach of. Everything a view finds is brought back
// to the particle's own side before it is used.
//
// The `vessel` module says where the openings lead; it must define, beside what
// `particles.wgsl` asks of it:
//   VESSEL_OPENINGS                              how many openings it may have
//   vessel_through(p, opening, reach) -> Through a point within reach of an opening, beyond it
//   vessel_gone_through(p) -> Through            a point sunk into an opening, out of the other
//   vessel_turned(v, opening) -> vec3<f32>       a vector carried through an opening
//   vessel_apart(brought, other) -> vec4<f32>    from a point beyond an opening to one brought
//                                                through it, as the side it was brought from has
//                                                it, with 1 where the way between them is open
// A vessel without openings has none, and its points are never through any.
#define_import_path fluid_views
#import vessel::{Through, VESSEL_OPENINGS, vessel_through, vessel_turned, vessel_apart}

const VIEWS: u32 = 1u + VESSEL_OPENINGS;

struct View {
    // where the particle is in this view, and whether it is there at all
    q: vec3<f32>,
    there: bool,
    // 0 for the particle's own view, or one more than the opening this one is through
    through: u32,
    beyond: Through,
}

/// The `n`th view from a point of what is within `reach` of it.
fn view_of(q: vec3<f32>, n: u32, reach: f32) -> View {
    if (n == 0u) {
        return View(q, true, 0u, Through(q, false, 0u, vec3(0.0)));
    }
    let beyond = vessel_through(q, n - 1u, reach);
    return View(beyond.p, beyond.there, n, beyond);
}

/// From a neighbour a view found to the particle, as the particle has it, with 1 where the two
/// are next to each other in that view at all.
fn apart(view: View, neighbour: vec3<f32>) -> vec4<f32> {
    if (view.through == 0u) {
        return vec4(view.q - neighbour, 1.0);
    }
    return vessel_apart(view.beyond, neighbour);
}

/// A vector of a neighbour a view found, as the particle has it.
fn brought_back(view: View, v: vec3<f32>) -> vec3<f32> {
    if (view.through == 0u) {
        return v;
    }
    return vessel_turned(v, 2u - view.through);
}
