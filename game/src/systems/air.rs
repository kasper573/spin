//! The air the ring holds, which is what it has in place of a sky.
//!
//! Air is kept as a state rather than as a set of finished coefficients: a composition, a
//! temperature, the pressure it stands at where it meets the ground, and what it carries in
//! suspension. Everything the renderer needs is worked out from those, so what the ring looks
//! like follows from what its air is rather than from numbers chosen to make it look right.
//! What comes out:
//!
//! - Scattering off the molecules themselves, from the Rayleigh cross-section, which goes as
//!   the inverse fourth power of the wavelength and as the number density. Blue skies are what
//!   that power law looks like over a long enough path; over thirty metres of habitat air it is
//!   a twentieth of one percent, and the same law says so.
//! - Scattering off what the air carries, from the size of the grains and how much of them
//!   there is. Grains much wider than a wavelength take every colour alike and throw the light
//!   forward, which is why fog and cloud are white and why a dusty beam shows; grains much
//!   narrower behave as the molecules do. Both fall out of the same expression.
//! - How much the air slows light, and how much more it slows blue than red, which is what
//!   bends a ray crossing a density gradient and spreads it into colours at a grazing angle.
//! - How the density falls away from the rim. A ring holds its air by spinning, so its air
//!   thins toward the axis exactly as a planet's thins with height, and by the same law: the
//!   potential is the one the spin makes.
use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;
use num_complex::Complex64;

use crate::core::units::{
    Kelvin, KilogramsPerCubicMetre, Metres, Nanometres, Pascals, RadiansPerSecond,
};

/// Boltzmann's constant, the molar gas constant, and the wavelengths the eye's three kinds of
/// cone answer to most strongly, which are what the three numbers of a colour stand for.
const BOLTZMANN: f64 = 1.380649e-23;
const MOLAR_GAS: f64 = 8.314462618;
pub const CHANNELS: [Nanometres; 3] = [Nanometres(612.0), Nanometres(549.0), Nanometres(465.0)];

/// What a gas is made of: the share of its molecules each kind makes up, that kind's molar
/// mass, and how much its molecules depart from a sphere, which is what makes them scatter a
/// little more than a sphere would and is measured as a depolarisation.
#[derive(Clone, Copy, Debug)]
pub struct Gas {
    pub share: f64,
    pub molar_mass: f64,
    pub depolarisation: Depolarisation,
}

/// How a kind of molecule's depolarisation runs with wavelength, as the fits to the measured
/// values do: a constant, a term in the inverse square of the wavelength in micrometres, and
/// one in the inverse fourth.
#[derive(Clone, Copy, Debug)]
pub struct Depolarisation {
    pub flat: f64,
    pub square: f64,
    pub quartic: f64,
}

impl Depolarisation {
    fn at(&self, wavelength: Nanometres) -> f64 {
        let micrometres = wavelength.0 / 1000.0;
        let inverse = 1.0 / (micrometres * micrometres);
        self.flat + self.square * inverse + self.quartic * inverse * inverse
    }
}

/// Dry air as a planet makes it, which is what a habitat would fill itself with.
pub const NITROGEN: Gas = Gas {
    share: 0.78084,
    molar_mass: 0.0280134,
    depolarisation: Depolarisation {
        flat: 1.034,
        square: 3.17e-4,
        quartic: 0.0,
    },
};
pub const OXYGEN: Gas = Gas {
    share: 0.20946,
    molar_mass: 0.0319988,
    depolarisation: Depolarisation {
        flat: 1.096,
        square: 1.385e-3,
        quartic: 1.448e-4,
    },
};
pub const ARGON: Gas = Gas {
    share: 0.00934,
    molar_mass: 0.039948,
    depolarisation: Depolarisation {
        flat: 1.0,
        square: 0.0,
        quartic: 0.0,
    },
};
pub const CARBON_DIOXIDE: Gas = Gas {
    share: 0.00036,
    molar_mass: 0.0440095,
    depolarisation: Depolarisation {
        flat: 1.15,
        square: 0.0,
        quartic: 0.0,
    },
};

/// What the air carries in suspension: how much of it there is by mass in a cubic metre, how
/// wide a grain of it is, how dense the stuff of a grain is, and how that stuff answers light —
/// how much it slows it, and how much of it a grain swallows rather than turns aside.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Suspension {
    pub loading: KilogramsPerCubicMetre,
    pub radius: Metres,
    pub density: KilogramsPerCubicMetre,
    pub slowing: f64,
    pub swallowing: f64,
}

/// What a grain does to light of one wavelength: the share of its own shadow's worth of light
/// it takes out of a beam, the share of that it turns aside rather than swallows, and how far
/// forward it throws what it turns.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scattered {
    pub taken: f64,
    pub turned: f64,
    pub forward: f64,
}

impl Suspension {
    /// The fine dust a filtered habitat still carries: a hundredth of a gram in a thousand
    /// cubic metres of it, in grains a third of a micrometre across, of the stuff of rock.
    pub const DUST: Suspension = Suspension {
        loading: KilogramsPerCubicMetre(1.0e-8),
        radius: Metres(0.3e-6),
        density: KilogramsPerCubicMetre(1500.0),
        slowing: 0.55,
        swallowing: 0.0,
    };

    /// Nothing carried at all: the air's molecules and no more.
    pub const CLEAR: Suspension = Suspension {
        loading: KilogramsPerCubicMetre(0.0),
        ..Suspension::DUST
    };

    /// A fog: droplets of water several micrometres across, wider than any wavelength of light,
    /// so that every colour is turned alike and the fog shows white.
    pub const FOG: Suspension = Suspension {
        loading: KilogramsPerCubicMetre(2.0e-5),
        radius: Metres(8.0e-6),
        density: KilogramsPerCubicMetre(1000.0),
        slowing: 0.333,
        swallowing: 0.0,
    };

    /// Smoke: grains a tenth the size of a wavelength, of soot, which swallows most of what it
    /// takes rather than turning it aside.
    pub const SMOKE: Suspension = Suspension {
        loading: KilogramsPerCubicMetre(1.0e-7),
        radius: Metres(0.05e-6),
        density: KilogramsPerCubicMetre(1800.0),
        slowing: 0.95,
        swallowing: 0.79,
    };

    /// How many grains stand in a cubic metre, given that each is a sphere of this stuff.
    pub fn count(&self) -> f64 {
        let r = self.radius.0 as f64;
        let grain = 4.0 / 3.0 * std::f64::consts::PI * r * r * r * self.density.0;
        if grain <= 0.0 {
            return 0.0;
        }
        self.loading.0 / grain
    }

    /// What a grain does to light of this wavelength, from Mie's solution of Maxwell's
    /// equations for a sphere: the series is summed as Bohren and Huffman set it out, with the
    /// logarithmic derivative inside the grain found by running its recurrence downward, which
    /// is the only way it is stable. Nothing here is fitted or capped — a grain much narrower
    /// than the wavelength comes out taking a share going as the inverse fourth power of it,
    /// which is why fine smoke is blue, and one much wider comes out taking twice its own
    /// area's worth of every colour alike and throwing it forward, which is why fog is white.
    pub fn scattering(&self, wavelength: Nanometres) -> Scattered {
        let metres = wavelength.0 * 1e-9;
        let x = 2.0 * std::f64::consts::PI * self.radius.0 as f64 / metres;
        if x <= 0.0 || self.count() <= 0.0 {
            return Scattered::default();
        }
        let m = Complex64::new(self.slowing + 1.0, self.swallowing);
        mie(x, m)
    }

    /// What a metre of the suspension takes out of a ray at this wavelength, and what it turns
    /// aside: the grains' shadows, each counted for as much as Mie's solution says.
    pub fn extinction(&self, wavelength: Nanometres) -> f64 {
        let r = self.radius.0 as f64;
        self.count() * std::f64::consts::PI * r * r * self.scattering(wavelength).taken
    }

    pub fn scattered(&self, wavelength: Nanometres) -> f64 {
        let r = self.radius.0 as f64;
        self.count() * std::f64::consts::PI * r * r * self.scattering(wavelength).turned
    }

    /// How far forward a grain throws what it scatters, as the one number a Henyey-Greenstein
    /// lobe takes.
    pub fn lobe(&self, wavelength: Nanometres) -> f64 {
        self.scattering(wavelength).forward
    }
}

/// Mie's solution for a sphere of size parameter `x` whose stuff answers light as `m` does
/// against what surrounds it, summed as `bhmie` does: `taken` and `turned` are the extinction
/// and scattering efficiencies, `forward` the asymmetry of what is scattered.
fn mie(x: f64, m: Complex64) -> Scattered {
    let terms = (x + 4.0 * x.cbrt() + 2.0).ceil() as usize;
    let mx = m * x;
    // the logarithmic derivative of the Riccati-Bessel function inside the grain, run downward
    // from well past the last term that matters, where it may be started at nothing
    let down = terms.max(mx.norm().ceil() as usize) + 15;
    let mut d = vec![Complex64::new(0.0, 0.0); down + 1];
    for n in (1..=down).rev() {
        let over = Complex64::new(n as f64, 0.0) / mx;
        d[n - 1] = over - 1.0 / (d[n] + over);
    }
    // the Riccati-Bessel functions outside it, run upward, where that is the stable way
    let (mut psi_back, mut psi_here) = (x.cos(), x.sin());
    let (mut chi_back, mut chi_here) = (-x.sin(), x.cos());
    let mut xi_back = Complex64::new(psi_here, -chi_here);
    let (mut taken, mut turned) = (0.0, 0.0);
    let mut along = Complex64::new(0.0, 0.0);
    let mut across = Complex64::new(0.0, 0.0);
    let (mut a_back, mut b_back) = (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    for (n, &slope) in (1..=terms).zip(&d[1..=terms]) {
        let order = n as f64;
        let psi = (2.0 * order - 1.0) / x * psi_here - psi_back;
        let chi = (2.0 * order - 1.0) / x * chi_here - chi_back;
        let xi = Complex64::new(psi, -chi);
        let over = order / x;
        let a = ((slope / m + over) * psi - psi_here) / ((slope / m + over) * xi - xi_back);
        let b = ((slope * m + over) * psi - psi_here) / ((slope * m + over) * xi - xi_back);
        taken += (2.0 * order + 1.0) * (a + b).re;
        turned += (2.0 * order + 1.0) * (a.norm_sqr() + b.norm_sqr());
        across +=
            Complex64::new((2.0 * order + 1.0) / (order * (order + 1.0)), 0.0) * (a * b.conj());
        if n > 1 {
            let last = order - 1.0;
            along += Complex64::new(last * (last + 2.0) / (last + 1.0), 0.0)
                * (a_back * a.conj() + b_back * b.conj());
        }
        (a_back, b_back) = (a, b);
        (psi_back, psi_here) = (psi_here, psi);
        (chi_back, chi_here) = (chi_here, chi);
        xi_back = xi;
    }
    let over = 2.0 / (x * x);
    let (taken, turned) = (over * taken, over * turned);
    Scattered {
        taken,
        turned,
        forward: if turned > 0.0 {
            4.0 / (x * x * turned) * (along.re + across.re)
        } else {
            0.0
        },
    }
}

/// The air a ring holds: what it is made of, how warm it is, what it presses at where it meets
/// the ground, and what it carries.
#[derive(Resource, Clone, Copy, Debug)]
pub struct Air {
    pub gases: [Gas; 4],
    pub temperature: Kelvin,
    /// The pressure where the air meets the ground, which is the furthest out the spin takes it.
    pub pressure: Pascals,
    pub carries: Suspension,
}

impl Default for Air {
    fn default() -> Self {
        Air {
            gases: [NITROGEN, OXYGEN, ARGON, CARBON_DIOXIDE],
            temperature: Kelvin(288.15),
            pressure: Pascals(101325.0),
            carries: Suspension::DUST,
        }
    }
}

impl Air {
    /// How many molecules stand in a cubic metre of it where it meets the ground.
    pub fn molecules(&self) -> f64 {
        if self.temperature.0 <= 0.0 {
            return 0.0;
        }
        self.pressure.0 / (BOLTZMANN * self.temperature.0)
    }

    /// What a mole of it weighs, and what a kilogram of it does per kelvin, which is what sets
    /// how fast it thins away from the rim.
    pub fn molar_mass(&self) -> f64 {
        let total: f64 = self.gases.iter().map(|g| g.share).sum();
        if total <= 0.0 {
            return 0.0;
        }
        self.gases
            .iter()
            .map(|g| g.share * g.molar_mass)
            .sum::<f64>()
            / total
    }

    /// How much less dense the air is at a distance from the axis than at the rim: a ring holds
    /// its air by spinning, so the potential the air stands in is the spin's, and an isothermal
    /// gas in it settles as `exp(w^2 (r^2 - rim^2) / 2 R T)`. This is the coefficient of
    /// `r^2 - rim^2` in that, per square metre.
    pub fn thinning(&self, spin: RadiansPerSecond) -> f64 {
        let specific = self.specific_gas_constant();
        if specific <= 0.0 || self.temperature.0 <= 0.0 {
            return 0.0;
        }
        let w = spin.0 as f64;
        w * w / (2.0 * specific * self.temperature.0)
    }

    pub fn specific_gas_constant(&self) -> f64 {
        let molar = self.molar_mass();
        if molar <= 0.0 { 0.0 } else { MOLAR_GAS / molar }
    }

    /// How much the air slows light at the rim, as `n - 1`. The molecules' response to light is
    /// what it is per molecule, so this follows the number density; the standard dispersion of
    /// air says how it runs with wavelength, and it is scaled from the density that dispersion
    /// was measured at to this one.
    pub fn slowing(&self, wavelength: Nanometres) -> f64 {
        standard_slowing(wavelength) * self.molecules() / STANDARD_MOLECULES
    }

    /// What a metre of the air at the rim scatters off its molecules at this wavelength, from
    /// the Rayleigh cross-section: it goes as the square of how much the gas slows light, as
    /// the inverse fourth power of the wavelength, and inversely as the number density, which
    /// together leave it going as the density once the slowing is taken as following it.
    pub fn rayleigh(&self, wavelength: Nanometres) -> f64 {
        let n = self.molecules();
        if n <= 0.0 {
            return 0.0;
        }
        let metres = wavelength.0 * 1e-9;
        let slowing = self.slowing(wavelength);
        // (n^2 - 1)^2 with n barely over one is four times (n - 1)^2
        let refraction = 4.0 * slowing * slowing;
        let pi = std::f64::consts::PI;
        8.0 * pi * pi * pi * refraction / (3.0 * n * metres.powi(4))
            * self.depolarisation(wavelength)
    }

    /// How much more the gas scatters than spheres would, weighted over what it is made of.
    pub fn depolarisation(&self, wavelength: Nanometres) -> f64 {
        let total: f64 = self.gases.iter().map(|g| g.share).sum();
        if total <= 0.0 {
            return 1.0;
        }
        self.gases
            .iter()
            .map(|g| g.share * g.depolarisation.at(wavelength))
            .sum::<f64>()
            / total
    }

    /// What a metre of it scatters off what it carries, at this wavelength, and what a metre
    /// of it takes out of a ray altogether, which is more where the grains swallow light.
    pub fn mie(&self, wavelength: Nanometres) -> f64 {
        self.carries.scattered(wavelength)
    }

    pub fn mie_extinction(&self, wavelength: Nanometres) -> f64 {
        self.carries.extinction(wavelength)
    }

    /// The whole of it as the shaders take it, for a ring turning this fast.
    pub fn uniform(&self, spin: RadiansPerSecond) -> AirUniform {
        let each = |f: &dyn Fn(Nanometres) -> f64| {
            Vec3::new(
                f(CHANNELS[0]) as f32,
                f(CHANNELS[1]) as f32,
                f(CHANNELS[2]) as f32,
            )
        };
        AirUniform {
            rayleigh: each(&|w| self.rayleigh(w)).extend(0.0),
            mie: each(&|w| self.mie(w)).extend(self.carries.lobe(CHANNELS[1]) as f32),
            taken: each(&|w| self.rayleigh(w) + self.mie_extinction(w)).extend(0.0),
            slowing: each(&|w| self.slowing(w)).extend(self.thinning(spin) as f32),
        }
    }
}

/// The air the standard dispersion was measured in: dry air at fifteen degrees and one
/// atmosphere, in molecules to the cubic metre.
const STANDARD_MOLECULES: f64 = 101325.0 / (BOLTZMANN * 288.15);

/// How much that standard air slows light, as `n - 1`, by the dispersion formula fitted to it.
fn standard_slowing(wavelength: Nanometres) -> f64 {
    let micrometres = wavelength.0 / 1000.0;
    let inverse = 1.0 / micrometres;
    let square = inverse * inverse;
    (8060.51 + 2480990.0 / (132.274 - square) + 17455.7 / (39.32957 - square)) * 1e-8
}

/// The air as the shaders take it: what a metre of it at the rim does to light, and how it
/// thins toward the axis. See [`Air`].
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct AirUniform {
    /// What a metre of it at the rim scatters off its molecules, per channel.
    pub rayleigh: Vec4,
    /// What a metre of it scatters off what it carries, per channel, and how far forward a
    /// grain of that throws it.
    pub mie: Vec4,
    /// What a metre of it takes out of a ray altogether, per channel: what it turns aside and
    /// what it swallows. Molecules swallow nothing, so this is over the scattering only by what
    /// the grains swallow, which for soot is most of what they take.
    pub taken: Vec4,
    /// How much it slows light at the rim, as `n - 1` per channel, and the coefficient of
    /// `r^2 - rim^2` in how its density falls away from the rim.
    pub slowing: Vec4,
}

impl Default for AirUniform {
    fn default() -> Self {
        Air::default().uniform(RadiansPerSecond(0.0))
    }
}
