//! Materials told something afresh every frame. Bevy makes a material's uniforms a new buffer
//! and bind group whenever the material changes, which costs the CPU dearly when every material
//! is told where it is seen from every frame. A buffer written in place costs nothing of the
//! kind, but Bevy binds those as storage, which the GPU reads far slower than uniforms from
//! fragment shaders that read them over and over. An `InPlace` material binds the buffers its
//! shaders read as uniforms, written in place.
//!
//! Bevy binds the images and buffers a material names as they are on the GPU when it makes the
//! material's bind group, and makes that again only when the material itself changes. An image
//! or a buffer is made anew on the GPU when its size or kind changes, so an `InPlace` material
//! is changed whenever one it names as a `#[dependency]` is made anew, lest it go on drawing
//! from what is no longer drawn into.
use std::hash::{DefaultHasher, Hash, Hasher};
use std::marker::PhantomData;

use bevy::asset::{AssetEventSystems, VisitAssetDependencies};
use bevy::ecs::system::SystemParamItem;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::render_resource::encase::internal::WriteInto;
use bevy::render::render_resource::encase::{self, ShaderType};
use bevy::render::render_resource::{
    AsBindGroup, AsBindGroupError, BindGroupLayout, BindGroupLayoutEntry, BindingType,
    BindlessDescriptor, BindlessSlabResourceLimit, BufferBindingType, BufferUsages,
    UnpreparedBindGroup,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::storage::ShaderBuffer;

/// A material whose `#[storage(n, read_only)]` buffers at `UNIFORMS` hold uniforms, written in
/// place with [`Tell`], which its shaders read as `var<uniform>`.
pub trait InPlaceUniforms: AsBindGroup + Asset {
    const UNIFORMS: &'static [u32];
}

#[derive(Asset, TypePath, Clone, Default, Deref, DerefMut)]
pub struct InPlace<M: InPlaceUniforms>(#[dependency] pub M);

/// Draws with `InPlace<M>` materials, each made anew whenever an image or a buffer it binds is.
pub struct InPlacePlugin<M>(PhantomData<M>);

impl<M> Default for InPlacePlugin<M> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<M: InPlaceUniforms> Plugin for InPlacePlugin<M>
where
    InPlace<M>: Material,
    <InPlace<M> as AsBindGroup>::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<InPlace<M>>::default())
            .add_systems(PostUpdate, rebind::<M>.before(AssetEventSystems));
    }
}

/// A buffer that holds a uniform, and is told it anew.
pub trait Tell {
    fn uniform<T: ShaderType + WriteInto>(value: &T) -> Self;
    fn tell<T: ShaderType + WriteInto>(&mut self, value: &T);
}

impl Tell for ShaderBuffer {
    fn uniform<T: ShaderType + WriteInto>(value: &T) -> Self {
        let mut buffer = ShaderBuffer::default();
        buffer.buffer_description.usage |= BufferUsages::UNIFORM;
        buffer.tell(value);
        buffer
    }

    fn tell<T: ShaderType + WriteInto>(&mut self, value: &T) {
        let mut bytes = encase::UniformBuffer::new(Vec::new());
        bytes
            .write(value)
            .expect("a uniform fits in the memory it is written to");
        self.data = Some(bytes.into_inner());
    }
}

impl<M: InPlaceUniforms> AsBindGroup for InPlace<M> {
    type Data = M::Data;
    type Param = M::Param;

    fn bindless_slot_count() -> Option<BindlessSlabResourceLimit> {
        M::bindless_slot_count()
    }

    fn bindless_supported(render_device: &RenderDevice) -> bool {
        M::bindless_supported(render_device)
    }

    fn label() -> &'static str {
        M::label()
    }

    fn bind_group_data(&self) -> Self::Data {
        self.0.bind_group_data()
    }

    fn unprepared_bind_group(
        &self,
        layout: &BindGroupLayout,
        render_device: &RenderDevice,
        param: &mut SystemParamItem<'_, '_, Self::Param>,
        force_no_bindless: bool,
    ) -> Result<UnpreparedBindGroup, AsBindGroupError> {
        self.0
            .unprepared_bind_group(layout, render_device, param, force_no_bindless)
    }

    fn bind_group_layout_entries(
        render_device: &RenderDevice,
        force_no_bindless: bool,
    ) -> Vec<BindGroupLayoutEntry> {
        let mut entries = M::bind_group_layout_entries(render_device, force_no_bindless);
        for entry in &mut entries {
            if M::UNIFORMS.contains(&entry.binding)
                && let BindingType::Buffer { ty, .. } = &mut entry.ty
            {
                *ty = BufferBindingType::Uniform;
            }
        }
        entries
    }

    fn bindless_descriptor() -> Option<BindlessDescriptor> {
        M::bindless_descriptor()
    }
}

/// Change every material some image or buffer it binds has been made anew for since it last
/// looked, so that Bevy binds what is there now.
fn rebind<M: InPlaceUniforms>(
    mut bound: Local<HashMap<AssetId<InPlace<M>>, u64>>,
    images: Res<Assets<Image>>,
    buffers: Res<Assets<ShaderBuffer>>,
    mut materials: ResMut<Assets<InPlace<M>>>,
) {
    let remade: Vec<_> = materials
        .iter()
        .filter_map(|(id, material)| {
            let now = made(material, &images, &buffers);
            bound
                .insert(id, now)
                .is_some_and(|was| was != now)
                .then_some(id)
        })
        .collect();
    for id in remade {
        if let Some(material) = materials.get_mut(id) {
            material.into_inner();
        }
    }
}

/// What the images and buffers a material binds are made as on the GPU: they are made anew
/// when this changes.
fn made(
    material: &impl VisitAssetDependencies,
    images: &Assets<Image>,
    buffers: &Assets<ShaderBuffer>,
) -> u64 {
    let mut made = DefaultHasher::new();
    material.visit_dependencies(&mut |id| {
        if let Ok(id) = id.try_typed::<Image>() {
            images
                .get(id)
                .map(|image| &image.texture_descriptor)
                .hash(&mut made);
        } else if let Ok(id) = id.try_typed::<ShaderBuffer>() {
            buffers
                .get(id)
                .map(|buffer| {
                    (
                        &buffer.buffer_description,
                        buffer.data.as_ref().map(Vec::len),
                    )
                })
                .hash(&mut made);
        }
    });
    made.finish()
}
