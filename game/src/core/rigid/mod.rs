//! Rigid bodies: shared shapes, per-body state, and impulse-based contacts.
mod body;
mod contacts;

pub use body::{Body, BodyShape, Collider};
pub use contacts::{collide_pair, collide_vessel};
