//! What the solid things in the ring are made of: the standard lit material, and with it the
//! light that reaches them through a pair of portals, which the standard lights know nothing
//! of. A material holds the mouths as the vantage it is drawn for has them, so whatever is
//! echoed for another vantage is drawn in a copy of its material made for that one.
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;

use crate::core::in_place::{InPlace, InPlaceUniforms};
use crate::systems::portal::Mouths;
use crate::systems::scene::VANTAGES;

pub type SolidMaterial = ExtendedMaterial<StandardMaterial, InPlace<LitThroughPortals>>;

pub fn solid(base: StandardMaterial) -> SolidMaterial {
    SolidMaterial {
        base,
        extension: InPlace::default(),
    }
}

/// The bindings `portals.wgsl` reads the mouths from where `MOUTHS_ON_A_SOLID` is defined,
/// clear of the standard material's own. Nothing is seen through a mouth in a solid thing, so
/// the pictures are left out.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct LitThroughPortals {
    #[storage(100, read_only)]
    mouths: Handle<ShaderBuffer>,
    #[texture(101)]
    #[sampler(103)]
    through_blue: Option<Handle<Image>>,
    #[texture(102)]
    through_orange: Option<Handle<Image>>,
    #[texture(104)]
    through_blue_other: Option<Handle<Image>>,
    #[texture(105)]
    through_orange_other: Option<Handle<Image>>,
}

impl InPlaceUniforms for LitThroughPortals {
    const UNIFORMS: &'static [u32] = &[100];
}

impl MaterialExtension for InPlace<LitThroughPortals> {
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
            .add_systems(PostUpdate, light);
    }
}

/// The copies of each solid material that the vantages beyond the mouths draw it in.
#[derive(Resource, Default)]
pub struct SolidCopies {
    of: HashMap<AssetId<SolidMaterial>, [Handle<SolidMaterial>; VANTAGES - 1]>,
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

    fn is_copy(&self, id: AssetId<SolidMaterial>) -> bool {
        self.of.values().flatten().any(|copy| copy.id() == id)
    }
}

/// Give every solid material the mouths of the vantage it is drawn for, and make its copies
/// anew whenever it changes.
fn light(
    mouths: Res<Mouths>,
    copies: Res<SolidCopies>,
    mut changes: MessageReader<AssetEvent<SolidMaterial>>,
    mut materials: ResMut<Assets<SolidMaterial>>,
) {
    let changed: HashSet<AssetId<SolidMaterial>> = changes
        .read()
        .filter_map(|change| match change {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => Some(*id),
            _ => None,
        })
        .collect();
    let originals: Vec<_> = materials.ids().filter(|id| !copies.is_copy(*id)).collect();
    for id in originals {
        hold_mouths(&mut materials, id, mouths.seen_from(0));
        let Some(made) = copies.of.get(&id) else {
            continue;
        };
        let base = changed
            .contains(&id)
            .then(|| materials.get(id).map(|original| original.base.clone()))
            .flatten();
        for (k, copy) in made.iter().enumerate() {
            if let Some(base) = &base
                && let Some(mut material) = materials.get_mut(copy)
            {
                material.base = base.clone();
            }
            hold_mouths(&mut materials, copy.id(), mouths.seen_from(k + 1));
        }
    }
}

fn hold_mouths(
    materials: &mut Assets<SolidMaterial>,
    id: AssetId<SolidMaterial>,
    mouths: Handle<ShaderBuffer>,
) {
    if let Some(mut material) = materials.get_mut(id)
        && material.extension.mouths != mouths
    {
        material.extension.mouths = mouths;
    }
}
