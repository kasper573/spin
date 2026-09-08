//! Rigid boxes: a shared shape, per-body state, and impulse-based contacts.
mod body;
mod contacts;

pub use body::{Body, BoxShape};
pub use contacts::{collide_pair, collide_vessel};
