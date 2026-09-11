//! The desktop window the native client runs in: it has no page, so there is no canvas to keep
//! in step with, no storage between runs, nothing to drop the pointer lock and no script hooks.
use bevy::prelude::*;

pub fn primary_window() -> Window {
    Window {
        title: "Spin-gravity wheel".to_owned(),
        ..default()
    }
}

pub fn storage_open() {}

pub fn storage_opened() -> Option<Vec<(String, String)>> {
    Some(Vec::new())
}

pub fn storage_write(_clear: bool, _puts: &[(String, String)], _deletes: &[String]) {}

pub fn storage_committed() -> u32 {
    0
}

pub fn storage_failed() -> u32 {
    0
}

pub fn sync_window(_window: &mut Window) {}

pub fn pointer_locked() -> bool {
    true
}

pub fn install_script_hooks() {}

pub fn take_script_commands() -> Vec<String> {
    Vec::new()
}

pub fn publish_status(_status: &str) {}
