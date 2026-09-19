//! The pair of portals: where their mouths may stand, and what going through them does.
use game::core::math::{Vec3d, cross, dot, norm, quat_rotate};
use game::core::units::Metres;
use game::core::vessel::Vessel;
use game::systems::drum::{
    CapSide, DEFAULT_RING, Drum, DrumSurface, FLAME_BAND, Mouth, MouthAnchor, MouthColour,
    MouthCoords, MouthFit, OPENING, Ring,
};

const BLUE: MouthColour = MouthColour::Blue;
const ORANGE: MouthColour = MouthColour::Orange;
const UP: Vec3d = [0.0, 1.0, 0.0];
const RIGHT: Vec3d = [0.0, 0.0, 1.0];

/// A mouth wanted on the ground `turn` round the ring from the site and `along` the axis.
fn on_the_ground(drum: &Drum, turn: f64, along: f64) -> Mouth {
    let at = drum.wall_point(turn, along - drum.site.y);
    drum.mouth_at(at, DrumSurface::Wall, UP, RIGHT)
}

/// A mouth wanted on the high cap, `turn` round the ring and `depth` in from the glass.
fn on_the_cap(drum: &Drum, turn: f64, depth: f64) -> Mouth {
    let half_width = drum.ring.half_width.0 as f64;
    let glass = drum.wall_point(turn, half_width - drum.site.y);
    let (_, outward) = drum.depth_and_outward(glass);
    let at = [
        glass[0] - outward[0] * depth,
        glass[1],
        glass[2] - outward[2] * depth,
    ];
    drum.mouth_at(at, DrumSurface::Cap(CapSide::High), [-1.0, 0.0, 0.0], RIGHT)
}

fn put(drum: &mut Drum, colour: MouthColour, wanted: Mouth, diameter: f32) -> MouthFit {
    let fit = drum.fit_mouth(colour, wanted, Metres(diameter));
    drum.put_mouth(colour, fit, Metres(diameter));
    fit
}

fn open(drum: &mut Drum) {
    drum.advance(2.0 * OPENING.0 as f64);
    assert_eq!(drum.mouths.fill(), 0.0, "a pair left to itself is open");
}

/// A drum with hills and hollows where the mouths of [`a_pair`] stand.
fn sculpted() -> Drum {
    let mut drum = Drum::new(DEFAULT_RING);
    for (k, (turn, along)) in [(0.0, 0.0), (0.03, 0.4), (1.2, -1.0), (1.17, -1.5)]
        .into_iter()
        .enumerate()
    {
        let at = drum.wall_point(turn, along);
        drum.sculpt(
            at,
            1.5,
            0.6 * (k as f64 + 1.0) * if k % 2 == 0 { 1.0 } else { -0.3 },
        );
    }
    drum
}

/// An open pair on the ground of a drum: blue by the site, orange a way round the ring.
fn a_pair(drum: &mut Drum) -> (Mouth, Mouth) {
    let blue = on_the_ground(drum, 0.0, 0.0);
    let orange = on_the_ground(drum, 1.2, -1.0);
    put(drum, BLUE, blue, 2.0);
    put(drum, ORANGE, orange, 2.0);
    open(drum);
    (
        *drum.mouths.get(BLUE).expect("blue stands"),
        *drum.mouths.get(ORANGE).expect("orange stands"),
    )
}

fn sub(a: Vec3d, b: Vec3d) -> Vec3d {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[test]
fn a_lone_portal_lets_nothing_through() {
    let mut drum = Drum::new(DEFAULT_RING);
    let wanted = on_the_ground(&drum, 0.0, 0.0);
    put(&mut drum, BLUE, wanted, 2.0);
    drum.advance(2.0 * OPENING.0 as f64);
    assert_eq!(drum.mouths.fill(), 1.0, "a mouth alone stays filled in");
    let blue = *drum.mouths.get(BLUE).expect("blue stands");
    let above = drum.mouth_point(
        &blue,
        MouthCoords {
            u: 0.0,
            v: 0.0,
            h: 0.5,
        },
    );
    let below = drum.mouth_point(
        &blue,
        MouthCoords {
            u: 0.0,
            v: 0.0,
            h: -0.1,
        },
    );
    assert_eq!(drum.passage(above, below), None);
}

#[test]
fn a_pair_opens_from_its_middle_outward_and_only_what_is_open_lets_through() {
    let mut drum = Drum::new(DEFAULT_RING);
    let (blue, _) = a_pair(&mut drum);
    let wanted = on_the_ground(&drum, 1.2, -1.0);
    put(&mut drum, ORANGE, wanted, 2.0);
    assert_eq!(drum.mouths.fill(), 1.0, "a pair just made is still shut");
    let mut last = 1.0;
    let inside = |v: f64, h: f64| MouthCoords { u: 0.0, v, h };
    for _ in 0..12 {
        drum.advance(OPENING.0 as f64 / 10.0);
        let fill = drum.mouths.fill();
        assert!(
            fill <= last,
            "the fill grew from {last} to {fill} while opening"
        );
        last = fill;
        let open_to = drum.mouths.passable();
        for v in [0.1, 0.5, 0.9] {
            let from = drum.mouth_point(&blue, inside(v, 0.3));
            let to = drum.mouth_point(&blue, inside(v, -0.05));
            assert_eq!(
                drum.passage(from, to).is_some(),
                v < open_to,
                "{v} m from the middle of a mouth open out to {open_to} m"
            );
        }
    }
    assert_eq!(last, 0.0);
}

#[test]
fn going_through_is_a_proper_turn_that_keeps_speed() {
    let mut drum = sculpted();
    let (blue, orange) = a_pair(&mut drum);
    for (u, v) in [(0.0, 0.0), (0.6, -0.3), (-0.5, 0.7)] {
        let at = MouthCoords { u, v, h: -0.2 };
        let from = drum.mouth_point(&blue, MouthCoords { h: 0.2, ..at });
        let passage = drum
            .passage(from, drum.mouth_point(&blue, at))
            .expect("the pair is open");
        let [right, up, off] = drum.mouth_frame(&blue, at);
        let out = MouthCoords { u: -u, v, h: 0.2 };
        let [far_right, far_up, far_off] = drum.mouth_frame(&orange, out);
        let near = |a: Vec3d, b: Vec3d| norm(&sub(a, b)) < 1e-9;
        assert!(
            near(passage.turned(off), far_off.map(|x| -x)),
            "what goes in comes out"
        );
        assert!(near(passage.turned(up), far_up), "and the same way up");
        assert!(near(passage.turned(right), far_right.map(|x| -x)));
        let turned = [right, up, off].map(|axis| passage.turned(axis));
        assert!(
            dot(&cross(&turned[0], &turned[1]), &turned[2]) > 0.999,
            "a right hand stays a right hand"
        );
        let velocity = [1.3, -4.0, 2.2];
        let after = quat_rotate(&passage.turn, &velocity);
        assert!((norm(&after) - norm(&velocity)).abs() < 1e-12);
    }
}

#[test]
fn nothing_arrives_inside_the_ground() {
    let mut drum = sculpted();
    let (blue, orange) = a_pair(&mut drum);
    for depth in [0.001, 0.3, 2.0] {
        for (u, v) in [(0.0, 0.0), (0.8, 0.2), (-0.3, -0.9), (0.2, 0.95)] {
            let from = drum.mouth_point(&blue, MouthCoords { u, v, h: 0.1 });
            let under = drum.mouth_point(&blue, MouthCoords { u, v, h: -depth });
            let out = drum.passage(from, under).expect("the pair is open").point;
            let over = drum.height_above_glass(out) - drum.ground(out);
            assert!(
                (over - depth).abs() < 1e-9,
                "{depth} m under the one ground came out {over} m over the other"
            );
            let there = drum.mouth_coords(&orange, out);
            assert!((there.u + u).abs() < 1e-9 && (there.v - v).abs() < 1e-9);
        }
    }
}

#[test]
fn going_through_and_back_is_where_it_started() {
    let mut drum = sculpted();
    let (blue, orange) = a_pair(&mut drum);
    let start = MouthCoords {
        u: 0.4,
        v: -0.5,
        h: 0.25,
    };
    let at = drum.mouth_point(&blue, start);
    // what stands before the blue mouth is, as the orange mouth has it, behind the orange one
    let behind_orange = MouthCoords {
        u: -start.u,
        v: start.v,
        h: -start.h,
    };
    let from = drum.mouth_point(
        &orange,
        MouthCoords {
            h: 0.1,
            ..behind_orange
        },
    );
    let back = drum
        .passage(from, drum.mouth_point(&orange, behind_orange))
        .expect("the pair is open");
    assert!(norm(&sub(back.point, at)) < 1e-9);
    let under = drum.mouth_point(
        &blue,
        MouthCoords {
            h: -start.h,
            ..start
        },
    );
    let there = drum.passage(at, under).expect("the pair is open");
    let v = [0.7, 0.1, -2.0];
    assert!(norm(&sub(back.turned(there.turned(v)), v)) < 1e-9);
}

#[test]
fn only_what_comes_from_the_room_goes_through() {
    let mut drum = Drum::new(DEFAULT_RING);
    let (blue, _) = a_pair(&mut drum);
    let under = drum.mouth_point(
        &blue,
        MouthCoords {
            u: 0.0,
            v: 0.0,
            h: -0.2,
        },
    );
    let deeper = drum.mouth_point(
        &blue,
        MouthCoords {
            u: 0.0,
            v: 0.0,
            h: -3.0,
        },
    );
    assert_eq!(drum.passage(deeper, under), None, "from outside the drum");
    let beside = drum.mouth_point(
        &blue,
        MouthCoords {
            u: 1.5,
            v: 0.0,
            h: -0.2,
        },
    );
    let above = drum.mouth_point(
        &blue,
        MouthCoords {
            u: 0.0,
            v: 0.0,
            h: 0.2,
        },
    );
    assert_eq!(
        drum.passage(above, beside),
        None,
        "into the ground beside the mouth"
    );
}

#[test]
fn a_mouth_with_room_stands_where_it_was_wanted() {
    let drum = Drum::new(DEFAULT_RING);
    for wanted in [on_the_ground(&drum, 0.3, 1.0), on_the_cap(&drum, -0.4, 5.0)] {
        assert_eq!(
            drum.fit_mouth(BLUE, wanted, Metres(2.0)),
            MouthFit::Fits {
                mouth: wanted,
                other: None
            }
        );
    }
}

#[test]
fn a_mouth_too_near_an_edge_is_moved_to_fit() {
    let drum = Drum::new(DEFAULT_RING);
    let half_width = drum.ring.half_width.0 as f64;
    let rim = 1.0 + FLAME_BAND;
    let MouthFit::Fits { mouth, .. } = drum.fit_mouth(
        BLUE,
        on_the_ground(&drum, 0.2, half_width - 0.1),
        Metres(2.0),
    ) else {
        panic!("there is room on the wall");
    };
    let MouthAnchor::Wall { along, .. } = mouth.anchor else {
        panic!("it was wanted on the wall");
    };
    assert!(
        (along - (half_width - rim)).abs() < 1e-9,
        "moved to {along}"
    );

    let MouthFit::Fits { mouth, .. } =
        drum.fit_mouth(BLUE, on_the_cap(&drum, 0.2, 0.6), Metres(2.0))
    else {
        panic!("there is room on the cap");
    };
    for p in drum.mouth_rim(&mouth, rim, 0.0) {
        let over = drum.height_above_glass(p) - drum.ground(p);
        assert!(
            over > -1e-9,
            "the rim of a mouth on the cap is {over} m over the ground"
        );
    }
}

#[test]
fn a_mouth_too_near_the_other_is_moved_off_it() {
    let mut drum = Drum::new(DEFAULT_RING);
    let blue = on_the_ground(&drum, 0.0, 0.0);
    put(&mut drum, BLUE, blue, 2.0);
    let fit = drum.fit_mouth(ORANGE, on_the_ground(&drum, 0.05, 0.3), Metres(2.0));
    let MouthFit::Fits { mouth, other } = fit else {
        panic!("there is room beside the blue mouth");
    };
    assert_eq!(other, Some(blue));
    let at = drum.mouth_coords(&blue, drum.mouth_point(&mouth, MouthCoords::default()));
    assert!(
        (at.across() - 2.0 * (1.0 + FLAME_BAND)).abs() < 1e-6,
        "the mouths' centres are {} m apart",
        at.across()
    );
    assert!(
        at.u > 0.0 && at.v > 0.0,
        "it was moved the way it already lay"
    );
}

#[test]
fn a_mouth_that_cannot_fit_is_refused() {
    let mut drum = Drum::new(DEFAULT_RING);
    let wide = Metres(2.0 * drum.ring.half_width.0);
    let wanted = on_the_ground(&drum, 0.0, 0.0);
    assert_eq!(
        drum.fit_mouth(BLUE, wanted, wide),
        MouthFit::Refused(wanted)
    );
    put(&mut drum, BLUE, wanted, wide.0);
    assert!(drum.mouths.is_empty(), "a refused mouth is not put");
}

#[test]
fn the_pair_shares_the_newest_diameter() {
    let mut drum = Drum::new(DEFAULT_RING);
    let half_width = drum.ring.half_width.0 as f64;
    let blue = on_the_ground(&drum, 0.0, half_width - 1.0 - FLAME_BAND);
    put(&mut drum, BLUE, blue, 2.0);
    let orange = on_the_ground(&drum, 1.5, 0.0);
    let fit = put(&mut drum, ORANGE, orange, 4.0);
    assert!(matches!(fit, MouthFit::Fits { .. }));
    assert_eq!(drum.mouths.diameter(), Metres(4.0));
    let MouthAnchor::Wall { along, .. } = drum.mouths.get(BLUE).expect("blue stands").anchor else {
        panic!("blue is on the wall");
    };
    assert!(
        (along - (half_width - 2.0 * (1.0 + FLAME_BAND))).abs() < 1e-9,
        "the standing mouth made room for its new size: {along}"
    );
}

#[test]
fn a_mouths_top_is_the_shooters_up_laid_on_the_surface() {
    let drum = Drum::new(DEFAULT_RING);
    let at = drum.wall_point(0.0, 0.0);
    for up in [[0.0, 1.0, 0.0], [0.0, 0.6, 0.8], [-0.7, -0.5, 0.5]] {
        let mouth = drum.mouth_at(at, DrumSurface::Wall, up, RIGHT);
        let [_, top, off] = drum.mouth_frame(&mouth, MouthCoords::default());
        let laid = sub(up, off.map(|x| x * dot(&up, &off)));
        let laid = laid.map(|x| x / norm(&laid));
        assert!(norm(&sub(top, laid)) < 1e-9, "{up:?} laid as {top:?}");
    }
    // looking straight down, up stands off the ground: the mouth is laid by the viewer's right
    let mouth = drum.mouth_at(at, DrumSurface::Wall, [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let [right, ..] = drum.mouth_frame(&mouth, MouthCoords::default());
    assert!(norm(&sub(right, [0.0, 1.0, 0.0])) < 1e-9);
}

#[test]
fn portals_follow_the_ground_sculpted_under_them() {
    let mut drum = Drum::new(DEFAULT_RING);
    let (blue, _) = a_pair(&mut drum);
    let before = drum.mouth_point(&blue, MouthCoords::default());
    drum.sculpt(before, 3.0, 1.0);
    let after = drum.mouth_point(&blue, MouthCoords::default());
    let rose = drum.height_above_glass(after) - drum.height_above_glass(before);
    assert!(
        (rose - 1.0).abs() < 1e-6,
        "the mouth rose {rose} m with a metre of ground"
    );
    assert_eq!(drum.mouths.get(BLUE), Some(&blue), "and is the same mouth");
}

#[test]
fn portals_keep_their_place_when_the_site_moves() {
    let mut drum = sculpted();
    let (blue, orange) = a_pair(&mut drum);
    let at = MouthCoords {
        u: 0.3,
        v: -0.4,
        h: 0.2,
    };
    let before = [blue, orange].map(|mouth| drum.mouth_point(&mouth, at));
    let shift = drum.resite(drum.wall_point(0.7, 2.0));
    for (mouth, before) in [blue, orange].into_iter().zip(before) {
        let moved = drum.mouth_point(&mouth, at);
        assert!(norm(&sub(moved, drum.carried(before, shift))) < 1e-9);
    }
}

#[test]
fn portals_survive_a_resize_or_close_where_they_no_longer_fit() {
    let mut drum = Drum::new(DEFAULT_RING);
    let (blue, _) = a_pair(&mut drum);
    let orange = on_the_ground(&drum, 1.2, 4.5);
    put(&mut drum, ORANGE, orange, 2.0);
    let from_site = |drum: &Drum, mouth: &Mouth| {
        let p = drum.mouth_point(mouth, MouthCoords::default());
        drum.turn_to(p) * drum.ring.radius.0 as f64
    };
    let arc = from_site(&drum, &blue);
    drum.resize(Ring {
        radius: Metres(40.0),
        half_width: Metres(3.0),
    });
    let blue = *drum.mouths.get(BLUE).expect("blue still fits");
    assert!(
        (from_site(&drum, &blue) - arc).abs() < 1e-6,
        "as far round from the site"
    );
    let MouthAnchor::Wall { along, .. } = drum.mouths.get(ORANGE).expect("orange fits").anchor
    else {
        panic!("orange is on the wall");
    };
    assert!(
        (along - (3.0 - 1.0 - FLAME_BAND)).abs() < 1e-9,
        "moved in from the nearer cap"
    );
    drum.resize(Ring {
        radius: Metres(40.0),
        half_width: Metres(1.0),
    });
    assert!(
        drum.mouths.is_empty(),
        "a drum narrower than the mouths holds none"
    );
}
