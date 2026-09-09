//! The desktop window the native client runs in: it has no page, so there is no canvas to keep
//! in step with, no storage between runs, nothing to drop the pointer lock and no script hooks.
use bevy::prelude::*;

pub fn primary_window() -> Window {
    Window {
        title: "Spin-gravity wheel".to_owned(),
        ..default()
    }
}

pub fn storage_load(_key: &str) -> Option<String> {
    None
}

pub fn storage_save(_key: &str, _value: &str) {}

pub fn sync_window(_window: &mut Window) {}

pub fn pointer_locked() -> bool {
    true
}

pub fn install_script_hooks() {}

pub fn take_script_commands() -> Vec<String> {
    Vec::new()
}

pub fn publish_status(_status: &str) {}
