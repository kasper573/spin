// The air the ring holds, which is what it has in place of a sky. Air scatters light out of a
// ray crossing it and scatters other light into it: off the molecules themselves, which take
// the short wavelengths far harder than the long ones and send them almost as readily backward
// as forward, and off what the air carries — dust, spray, smoke — whose grains are wider than
// a wavelength, take every colour much alike, and throw the light forward in a tight lobe.
//
// A habitat is not a planet. Thirty metres of air at the pressure a lung wants scatters about
// a twentieth of one percent of the light crossing it off its molecules, so nothing in the
// ring is a wavelength away from the colour it would have in vacuum; what does show is the
// forward lobe off the dust, as a faint bloom on the air between the eye and the sun. Air also
// slows light, which is why the water and the glass are given their indices against air rather
// than against vacuum; it bends light too, but the bending that shows on a planet — a flattened
// sun, a green flash — is a ray grazing hundreds of kilometres of thinning air, and the whole
// ring is thirty metres across.
#define_import_path air

const PI: f32 = 3.14159265;

/// What a metre of air at a habitat's pressure scatters off its molecules, at the wavelengths
/// the eye's three kinds of cone answer to. Scattering goes as the inverse fourth power of the
/// wavelength, which is the whole of why a sky is blue.
const RAYLEIGH: vec3<f32> = vec3<f32>(5.8e-6, 13.5e-6, 33.1e-6);
/// What a metre of it scatters off the dust and spray it carries, which is nearly the same at
/// every wavelength, and how tightly forward that scattering throws the light.
const MIE: f32 = 21e-6;
const MIE_LOBE: f32 = 0.76;
/// Air's refractive index: what the water and the glass are seen through it against.
const AIR_IOR: f32 = 1.000293;

/// What a metre of air takes out of a ray crossing it, whichever way it is turned aside.
fn air_extinction() -> vec3<f32> {
    return RAYLEIGH + vec3(MIE);
}

/// How readily the molecules turn light through this angle: as willingly backward as forward,
/// and half as willingly side on.
fn rayleigh_phase(cosine: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + cosine * cosine);
}

/// How readily the dust turns it: forward, overwhelmingly.
fn mie_phase(cosine: f32) -> f32 {
    let g = MIE_LOBE;
    let d = 1.0 + g * g - 2.0 * g * cosine;
    return (1.0 - g * g) / (4.0 * PI * d * sqrt(max(d, 1e-6)));
}

/// What a metre of air turns into a ray running along `dir`, out of a sun's light `sun` coming
/// from `to_sun`: off the molecules, which favour neither way much, and off the dust, which
/// throws it forward and so shows as a bloom on the air between the eye and the sun.
fn air_turned(dir: vec3<f32>, to_sun: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let cosine = dot(dir, to_sun);
    return (RAYLEIGH * rayleigh_phase(cosine) + MIE * mie_phase(cosine)) * sun;
}

/// What a metre of it turns into the ray out of light reaching it from everywhere at once,
/// which no phase favours a direction of.
fn air_lit_all_round(ambient: vec3<f32>) -> vec3<f32> {
    return air_extinction() * ambient / (4.0 * PI);
}

/// What `distance` metres of air do to the light `colour` that set out across them toward the
/// eye: what is left of it, and what the air turned into the ray on the way, given how much a
/// metre of it turns. The light turned in partway is dimmed by the air left in front of it,
/// which over the same stretch leaves the same share of it as of the light from behind.
fn through_air(colour: vec3<f32>, turned: vec3<f32>, distance: f32) -> vec3<f32> {
    let extinction = air_extinction();
    let left = exp(-extinction * distance);
    return colour * left + turned * (vec3(1.0) - left) / extinction;
}
