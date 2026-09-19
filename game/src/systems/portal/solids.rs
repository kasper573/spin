//! What the solid things in the ring are made of: the standard lit material, and with it the
//! light that reaches them through a pair of portals, which the standard lights know nothing
//! of. A material holds the mouths as the vantage it is drawn for has them, so whatever is
//! echoed for another vantage is drawn in a copy of its material made for that one.
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

use crate::core::units::Seconds;
use crate::systems::portal::MouthsUniform;
use crate::systems::scene::{SettleVantages, VANTAGES, Vantages};
use crate::systems::sim::Simulation;

pub type SolidMaterial = ExtendedMaterial<StandardMaterial, LitThroughPortals>;

pub fn solid(base: StandardMaterial) -> SolidMaterial {
    SolidMaterial {
        base,
        extension: LitThroughPortals::default(),
    }
}

/// The bindings `portals.wgsl` reads the mouths from where `MOUTHS_ON_A_SOLID` is defined,
/// clear of the standard material's own. Nothing is seen through a mouth in a solid thing, so
/// the pictures are left out.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct LitThroughPortals {
    #[uniform(100)]
    mouths: MouthsUniform,
    #[texture(101)]
    #[sampler(103)]
    through_blue: Option<Handle<Image>>,
    #[texture(102)]
    through_orange: Option<Handle<Image>>,
}

impl MaterialExtension for LitThroughPortals {
    fn fragment_shader() -> ShaderRef {
        "embedded://game/systems/shaders/solid.wgsl".into()
    }

    fn specialize(
        _: &bevy::pbr::MaterialExtensionPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        _: &bevy::mesh::MeshVertexBufferLayoutRef,
        _: bevy::pbr::MaterialExtensionKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        if let Some(fragment) = &mut descriptor.fragment {
            fragment.shader_defs.push("MOUTHS_ON_A_SOLID".into());
        }
        Ok(())
    }
}

pub struct SolidsPlugin;

impl Plugin for SolidsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<SolidMaterial>::default())
            .init_resource::<SolidCopies>()
            .add_systems(
                Update,
                light
                    .after(SettleVantages)
                    .in_set(crate::systems::sim::SimSet::Observe),
            );
    }
}

/// The copies of each solid material that the vantages beyond the mouths draw it in.
#[derive(Resource, Default)]
pub struct SolidCopies {
    of: HashMap<AssetId<SolidMaterial>, [Handle<SolidMaterial>; VANTAGES - 1]>,
    lit: bool,
}

impl SolidCopies {
    /// The material a vantage draws what is made of `material` in.
    pub fn seen_from(
        &mut self,
        material: &Handle<SolidMaterial>,
        vantage: usize,
        materials: &mut Assets<SolidMaterial>,
    ) -> Handle<SolidMaterial> {
        if vantage == 0 {
            return material.clone();
        }
        let copies = self.of.entry(material.id()).or_insert_with(|| {
            let copy = materials.get(material).cloned().unwrap_or_default();
            std::array::from_fn(|_| materials.add(copy.clone()))
        });
        copies[vantage - 1].clone()
    }
}

/// Keep every solid material's copies like it, and while a pair of portals is open, tell each
/// the mouths as its vantage has them.
fn light(
    sim: Res<Simulation>,
    vantages: Res<Vantages>,
    mut copies: ResMut<SolidCopies>,
    mut materials: ResMut<Assets<SolidMaterial>>,
) {
    let open = sim.drum.mouths.passable() > 0.0;
    let mouths: [Option<MouthsUniform>; VANTAGES] = std::array::from_fn(|k| {
        let vantage = vantages.0[k].as_ref()?;
        (open || copies.lit)
            .then(|| MouthsUniform::of(&sim.drum, vantage, [Mat4::ZERO; 2], Seconds(0.0)))
    });
    copies.lit = open;
    let originals: Vec<_> = materials
        .ids()
        .filter(|id| !copies.of.values().flatten().any(|copy| copy.id() == *id))
        .collect();
    for id in originals {
        let Some(original) = materials.get(id).cloned() else {
            continue;
        };
        if let Some(mouths) = &mouths[0]
            && let Some(mut material) = materials.get_mut(id)
        {
            material.extension.mouths = mouths.clone();
        }
        let Some(made) = copies.of.get(&id) else {
            continue;
        };
        for (k, copy) in made.iter().enumerate() {
            let Some(mut material) = materials.get_mut(copy) else {
                continue;
            };
            material.base = original.base.clone();
            if let Some(mouths) = &mouths[k + 1] {
                material.extension.mouths = mouths.clone();
            }
        }
    }
}
